# Gemma4 E2B

This folder contains the Gemma4 E2B accelerator inference. Both text-only
prompts and image + text (VLM) prompts are supported. The full pipeline —
patch embedder, 16-layer vision encoder, pooler, embed_vision projection,
35-layer language-model prefill, and decoder loop — runs on the accelerator.

## Layout

- **gemma4_e2b_test.py** – Compile + run. Builds `gemma4_instruction.bin` on first invocation (or extends it for VLM), then runs.
- **gemma4_e2b_run_from_bin.py** – **Execute-only.** Assumes the unified bin already exists; refuses to compile. Use this for fast demo / production runs after a one-time `gemma4_e2b_test.py` invocation.
- **gemma4_e2b_config.json** – Model and layout config.
- **gemma4_e2b_bin/** – Weights, HF model, and the single unified instruction bin. Contains:
  - `weights_gemma4_e2b_hf.bin` (generated from the HF model on first run)
  - `gemma-4-E2B-it/` (Hugging Face model, or set via config)
  - `gemma4_instruction.bin` + `gemma4_instruction.json` (the unified instruction bin: LM prefill/decode buckets + vision encoder ISA + vision rope pads — see "Single instruction bin" below and `notes/notes_gemma4_e2b.md` for the full schema)

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
| Max context | 1024 |
| Codec | IF4 (MixMSE_64) for matmul weights; BF16 for norms / scales / embeddings |

## Prerequisites

- Run from the **repo root directory** so that `user_dma_core` is on the path:
  ```bash
  python models/gemma4_e2b/gemma4_e2b_test.py
  ```
- Python with `torch`, `transformers >= 5.5.0`, and DMA device access.
- The HF model (`google/gemma-4-E2B-it`) is downloaded automatically into `gemma4_e2b_bin/gemma-4-E2B-it/` on first run (requires network access and a Hugging Face account with access to the Gemma license).

## Recommended first run: **VLM mode**

> **Run VLM on your first invocation, then use `gemma4_e2b_run_from_bin.py` for everything afterward.**

A first VLM run produces the complete unified instruction bin (`gemma4_instruction.bin`)
that holds **both** the LM prefill/decode buckets **and** the vision encoder ISA + rope pads.
Subsequent runs — VLM, LM-only, doesn't matter — load the same bin and skip
compilation entirely. Running LM-only first works too, but the bin is initially
LM-only and the first VLM invocation has to extend it (one-time ~80 s incremental
vision compile + atomic rewrite). Going VLM-first sidesteps that surprise.

```bash
# 1. One-time setup: builds LM core (~3.5 min) + extends with vision (~80 s)
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable

# 2. From now on use the execute-only script — no compile, just load + run.
#    The same bin serves every mode:
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py                                    # LM only
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --vision-enable                    # VLM, default image
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --image my_photo.jpg --prompt "?"  # VLM, different image
```

If you started with LM-only and now want VLM, just run
`gemma4_e2b_test.py --vision-enable` once — it detects the bin is missing its
vision section and incrementally extends it (LM bytes are reused as-is; only
vision is compiled fresh and appended). After that, `gemma4_e2b_run_from_bin.py`
works for any mode against the now-complete bin.

## Usage

From the **repo root** directory. Gemma4 E2B supports three modes — text only,
image + text (VLM), and audio + text. Exactly one encoder modality can be active
per run.

### Vision + text (VLM) — recommended initial run

```bash
# Use the shipped example image (../../test_samples/yosemite.jpg) with default prompt
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable

# Custom image and prompt
python models/gemma4_e2b/gemma4_e2b_test.py --image path/to/image.jpg --prompt "What is in this image?"
```

Images of any aspect ratio / resolution are accepted; the test pre-resizes them
to a canonical `896×896` so the vision encoder always sees a fixed
`num_patches = 2520`. See `notes_gemma4_e2b.md → "Vision encoder fixed-shape contract"`.

### Language model (text only)

```bash
# Default prompt
python models/gemma4_e2b/gemma4_e2b_test.py

# Custom prompt
python models/gemma4_e2b/gemma4_e2b_test.py --prompt "Your prompt here"
```

### Audio + text

```bash
# Use the shipped example audio (../../test_samples/apex.wav) with default prompt
python models/gemma4_e2b/gemma4_e2b_test.py --audio-enable

# Custom audio and prompt
python models/gemma4_e2b/gemma4_e2b_test.py --audio path/to/file.wav --prompt "Transcribe this exactly."
```

> **Note:** audio is not yet folded into `gemma4_instruction.bin`. It uses a
> separate side-cache (`audio_program_cache_*.bin`). The unified-bin extension
> for audio is planned and will mirror the vision incremental path.

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

## Single instruction bin

The script compiles **one** combined `gemma4_instruction.bin` containing:

- 8 prefill program buckets `seq_len ∈ {64, 128, 192, 256, 320, 384, 448, 512}`
- 16 decoder program buckets `seq_len ∈ {64, 128, ..., 1024}` (stride 64)
- 16-layer vision encoder one-shot ISA (after first VLM run)

Each section is dispatched at runtime by an absolute start address recorded in
the manifest (`gemma4_instruction.json`). PBI JUMP_ABS targets in every section
were baked against the manifest's `instruction_base_addr` (`0xA0000000`) plus
the per-section offset, so the bin must be DMA'd to that exact base address —
which `load_instruction_bin()` does automatically.

To force a clean rebuild (e.g. after changing buckets in the config or pulling
new ISA semantics):

```bash
rm gemma4_e2b_bin/gemma4_instruction.bin gemma4_e2b_bin/gemma4_instruction.json
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable    # rebuild everything
```

See `notes_gemma4_e2b.md` for the manifest schema, compile-version stamp, and
DRAM layout that keeps the bin under the 4 GB FPGA address ceiling.
