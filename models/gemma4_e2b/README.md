# Gemma4 E2B example

Gemma4 E2B accelerator inference. Three modes — text only (LM), image
+ text (VLM), audio + text — all run from a single instruction bin.

## Two scripts, two stages

| Stage | Script | Notes |
|---|---|---|
| Build (once) | `gemma4_e2b_test.py` | Builds `gemma4_instruction.bin` + `weights_gemma4_e2b_hf.bin`, then sanity-runs. Needs the HF model locally. One build covers all three modes. |
| Deploy (every run after) | `gemma4_e2b_run_from_bin.py` | Loads the bin files and runs. Never touches the HF model. |

After the first `gemma4_e2b_test.py` run, every subsequent invocation
(either script, any mode) skips compilation.

## Files

- **gemma4_e2b_test.py** – Build + run. Builds the unified bin and the
  combined weight bin on first invocation; otherwise just runs.
- **gemma4_e2b_run_from_bin.py** – Execute-only. Loads bin files,
  refuses to compile, no HF model on disk required.
- **gemma4_e2b_config.json** – Model + layout config.
- **gemma4_e2b_bin/** – On-disk artifacts:
  - `gemma4_instruction.bin` + `.json` – unified ISA (~1.1 GB):
    LM prefill + decode + vision encoder + audio encoder.
  - `weights_gemma4_e2b_hf.bin` + `.json` – combined weights (~7 GB)
    with sections: `[LM | vision | audio | host]`.
  - `tokenizer/` – minimal tokenizer + processor configs (~32 MB).

## Prerequisites

- Run from the **repo root** so `user_dma_core` is on the path.
- Python with `torch`, `transformers >= 5.5.0`, `Pillow`, `soundfile`
  (audio mode only), and FPGA device access via xdma.
- Build stage only: HF model `google/gemma-4-E2B-it` is downloaded
  automatically into `gemma4_e2b_bin/gemma-4-E2B-it/` on first run.
- Deploy stage: no HF model directory required.

## Build / sanity-check — `gemma4_e2b_test.py`

```bash
# VLM (recommended first run)
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable
python models/gemma4_e2b/gemma4_e2b_test.py --image my.jpg --prompt "?"

# LM only
python models/gemma4_e2b/gemma4_e2b_test.py
python models/gemma4_e2b/gemma4_e2b_test.py --prompt "What is 2+2?"

# Audio
python models/gemma4_e2b/gemma4_e2b_test.py --audio-enable
python models/gemma4_e2b/gemma4_e2b_test.py --audio clip.wav --prompt "Transcribe this."
```

The first run builds all three sections (~4 min: LM ~3.5 min +
vision ~80 s + audio ~17 s). Subsequent runs just load and execute.

Force a clean rebuild:

```bash
rm gemma4_e2b_bin/gemma4_instruction.bin gemma4_e2b_bin/gemma4_instruction.json
rm gemma4_e2b_bin/weights_gemma4_e2b_hf.bin gemma4_e2b_bin/weights_gemma4_e2b_hf.json
python models/gemma4_e2b/gemma4_e2b_test.py --vision-enable
```

## Deploy / demo — `gemma4_e2b_run_from_bin.py`

Execute-only. Refuses to compile, never imports the HF model class.

```bash
# LM
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --prompt "What is 2+2?"

# VLM
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --vision-enable
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --image my.jpg --prompt "?"

# Audio
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --audio-enable
python models/gemma4_e2b/gemma4_e2b_run_from_bin.py --audio clip.wav --prompt "Transcribe."
```

### Deploy footprint

```
gemma4_e2b_run_from_bin.py
gemma4_e2b_config.json
user_dma_core.py + quant_schemas.py
gemma4_e2b_bin/
  gemma4_instruction.bin + .json       (~1.1 GB)
  weights_gemma4_e2b_hf.bin  + .json   (~7 GB; LM + vision + audio + host sections)
  tokenizer/                           (~32 MB)
```

## Flags

- `--dev xdma0 --cycle 5.62` — DMA device and clock cycle (defaults).
- `--local-weights` — use the local full-model weights bin.
- `--image PATH` implies `--vision-enable`; `--vision-enable` alone
  uses the default image at `../../test_samples/yosemite.jpg`.
- `--audio PATH` implies `--audio-enable`; `--audio-enable` alone
  uses the default clip at `../../test_samples/apex.wav`.
- Vision and audio are mutually exclusive (exactly one modality per run).
