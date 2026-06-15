# Gemma4 E2B example

Gemma4 E2B accelerator inference. Three modes — text only (LM), image +
text (VLM), and audio + text — run from a single instruction bin.

## Two scripts, two stages

| Stage | Script | Notes |
|---|---|---|
| Build (once) | `gemma4_e2b_test.py` | Builds `gemma4_instruction.bin` + `weights_gemma4_e2b_hf.bin`, then sanity-runs. Needs the HF model locally. Default build includes LM + VLM + audio unless `GEMMA4_LM_ONLY_BIN=1`. |
| Deploy (every run after) | `gemma4_e2b_run_from_bin.py` | Loads the bin files and runs. Never touches the HF model. |

After the first `gemma4_e2b_test.py` run, every subsequent invocation
(either script, any mode present in the bin) skips compilation.

## Files

- **gemma4_e2b_test.py** – Build + run. Builds the unified bin and the
  combined weight bin on first invocation; otherwise just runs.
- **gemma4_e2b_run_from_bin.py** – Execute-only. Loads bin files,
  refuses to compile, no HF model on disk required.
- **gemma4_e2b_config.json** – Model + layout config.
- **gemma4_e2b_bin/** – On-disk artifacts:
  - `gemma4_instruction.bin` + `.json` – unified ISA. Full multimodal bin
    is 14.39 MiB on disk (15,086,688 B): LM prefill + decode + vision
    + audio encoders + vision RoPE tables. Used size by mode: LM-only
    5.78 MiB, VLM 7.24 MiB, audio 12.93 MiB.
  - `weights_gemma4_e2b_hf.bin` + `.json` – combined weights (~7 GB)
    with sections: `[LM | vision | audio | host]`.
  - `tokenizer/` – minimal tokenizer + processor configs (~32 MB).

## Prerequisites

- Run from the **repo root** so `user_dma_core` is on the path.
- Python with `torch`, `transformers >= 5.5.0`, `Pillow`,
  and FPGA device access via xdma.
- Build stage only: HF model `google/gemma-4-E2B-it` is downloaded
  automatically into `gemma4_e2b_bin/gemma-4-E2B-it/` on first run.
- Deploy stage: no HF model directory required.

## Build / sanity-check — `gemma4_e2b_test.py`

```bash
# VLM (recommended first run — builds vision encoder into the bin)
python src/template/models/gemma4_e2b/gemma4_e2b_test.py --vision-enable
python src/template/models/gemma4_e2b/gemma4_e2b_test.py --image my.jpg --prompt "?"

# Audio
python src/template/models/gemma4_e2b/gemma4_e2b_test.py --audio-enable
python src/template/models/gemma4_e2b/gemma4_e2b_test.py --audio my.wav --prompt "Describe this audio."

# LM only
python src/template/models/gemma4_e2b/gemma4_e2b_test.py
python src/template/models/gemma4_e2b/gemma4_e2b_test.py --prompt "What is 2+2?"
```

The first run builds the instruction bin and combined weight bin. Subsequent
runs just load and execute.

Force a clean rebuild:

```bash
rm gemma4_e2b_bin/gemma4_instruction.bin gemma4_e2b_bin/gemma4_instruction.json
rm gemma4_e2b_bin/weights_gemma4_e2b_hf.bin gemma4_e2b_bin/weights_gemma4_e2b_hf.json
python src/template/models/gemma4_e2b/gemma4_e2b_test.py --vision-enable
```

## Deploy / demo — `gemma4_e2b_run_from_bin.py`

Execute-only. Refuses to compile, never imports the HF model class.

```bash
# LM
python src/template/models/gemma4_e2b/gemma4_e2b_run_from_bin.py
python src/template/models/gemma4_e2b/gemma4_e2b_run_from_bin.py --prompt "What is 2+2?"

# VLM
python src/template/models/gemma4_e2b/gemma4_e2b_run_from_bin.py --vision-enable
python src/template/models/gemma4_e2b/gemma4_e2b_run_from_bin.py --image my.jpg --prompt "?"

# Audio
python src/template/models/gemma4_e2b/gemma4_e2b_run_from_bin.py --audio-enable
python src/template/models/gemma4_e2b/gemma4_e2b_run_from_bin.py --audio my.wav --prompt "Describe this audio."
```

### Deploy footprint

```
gemma4_e2b_run_from_bin.py
gemma4_e2b_config.json
user_dma_core.py + quant_lib.py
gemma4_e2b_bin/
  gemma4_instruction.bin + .json       (14.39 MiB full multimodal)
  weights_gemma4_e2b_hf.bin  + .json   (~7 GB; LM + vision + audio + host sections)
  tokenizer/                           (~32 MB)
```

## Flags

- `--dev xdma0 --cycle 5.62` — DMA device and clock cycle (defaults).
- `--local-weights` — use the local full-model weights bin.
- `--image PATH` implies `--vision-enable`; `--vision-enable` alone
  uses the default image at `../../test_samples/yosemite.jpg`.
- `--audio PATH` implies `--audio-enable`; `--audio-enable` alone
  uses the default audio at `../../test_samples/apex.wav`.
- `GEMMA4_LM_ONLY_BIN=1` during build skips the vision/audio instruction
  sections for a smaller LM-only artifact.

## Prompt-length limits (dynamic PBI)

The model ships as a single prefill + single decoder program
(`prefill_max_seq_len` and bucket-dispatched decoder; no per-bucket
ISA copies):

- **`prefill_max_seq_len` (config, default `512`)** — prompts up to
  this length are supported. Shorter prompts are padded at the host
  with the last real token; causal bias masks padded positions.
- **Decoder:** single captured program serves all context lengths
  up to `max_context_size` (default `1024`) via the
  `decoder_group_attention_core` bucket dispatcher.

Both knobs live in `gemma4_e2b_config.json`. Increasing
`prefill_max_seq_len` linearly grows the prefill ISA in
`gemma4_instruction.bin`; the decoder bin size is fixed.
