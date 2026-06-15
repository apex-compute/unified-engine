# Gemma4 E4B example

Gemma4 E4B accelerator inference from a single instruction bin. Text-only
(LM) is the default mode; image+text (VLM) and audio+text encoders are
opt-in (`GEMMA4_E4B_ALLOW_ENCODER=1` with `--vision-enable`/`--audio-enable`).

> **Note on E4B decode quality:** the LM and audio paths decode cleanly.
> The VLM path is image-dependent and can fall into repetition early or in
> the long tail (a documented IF4/depth-sensitivity issue in the LM, not a
> vision/ISA bug). Run encoder modes with `GEMMA4_PENALTY=1` to improve
> loop resistance, but do not treat E4B VLM quality as final.

## Two scripts, two stages

| Stage | Script | Notes |
|---|---|---|
| Build (once) | `gemma4_e4b_test.py` | Builds `gemma4_instruction.bin` + `weights_gemma4_e4b_hf.bin`, then sanity-runs. Needs the HF model locally. Default build is LM-only; set `GEMMA4_LM_ONLY_BIN=0` and `GEMMA4_E4B_ALLOW_ENCODER=1` to build/test encoder modes. |
| Deploy (every run after) | `gemma4_e4b_run_from_bin.py` | Loads the bin files and runs. Never touches the HF model. |

After the first `gemma4_e4b_test.py` run, every subsequent invocation
(either script) skips compilation.

## Files

- **gemma4_e4b_test.py** – Build + run. Builds the unified bin and the
  combined weight bin on first invocation; otherwise just runs.
- **gemma4_e4b_run_from_bin.py** – Execute-only. Loads bin files,
  refuses to compile, no HF model on disk required.
- **gemma4_e4b_config.json** – Model + layout config.
- **gemma4_e4b_bin/** – On-disk artifacts:
  - `gemma4_instruction.bin` + `.json` – instruction ISA. Full multimodal
    bin is 15.02 MiB on disk (15,746,912 B): LM prefill + decode + vision
    + audio encoders + vision RoPE tables. Used size by mode: LM-only
    6.38 MiB, VLM 7.84 MiB, audio 13.56 MiB.
  - `weights_gemma4_e4b_hf.bin` + `.json` – combined weights (~10 GB)
    with sections: `[LM | vision | audio | host]`.
  - `tokenizer/` – minimal tokenizer + processor configs (~32 MB).

## Prerequisites

- Run from the **repo root** so `user_dma_core` is on the path.
- Python with `torch`, `transformers >= 5.5.0`, and FPGA device access via xdma.
- Build stage only: HF model `google/gemma-4-E4B-it` is downloaded
  automatically into `gemma4_e4b_bin/gemma-4-E4B-it/` on first run.
- Deploy stage: no HF model directory required.

## Build / sanity-check — `gemma4_e4b_test.py`

```bash
python src/template/models/gemma4_e4b/gemma4_e4b_test.py
python src/template/models/gemma4_e4b/gemma4_e4b_test.py --prompt "What is 2+2?"

# Build/test encoder modes into the unified bin
GEMMA4_LM_ONLY_BIN=0 GEMMA4_E4B_ALLOW_ENCODER=1 \
  python src/template/models/gemma4_e4b/gemma4_e4b_test.py --vision-enable
GEMMA4_LM_ONLY_BIN=0 GEMMA4_E4B_ALLOW_ENCODER=1 \
  python src/template/models/gemma4_e4b/gemma4_e4b_test.py --audio-enable
```

First run builds the instruction bin and combined weight bin. Subsequent
runs just load and execute.

Force a clean rebuild:

```bash
rm gemma4_e4b_bin/gemma4_instruction.bin gemma4_e4b_bin/gemma4_instruction.json
rm gemma4_e4b_bin/weights_gemma4_e4b_hf.bin gemma4_e4b_bin/weights_gemma4_e4b_hf.json
python src/template/models/gemma4_e4b/gemma4_e4b_test.py
```

## Deploy / demo — `gemma4_e4b_run_from_bin.py`

Execute-only. Refuses to compile, never imports the HF model class.

```bash
python src/template/models/gemma4_e4b/gemma4_e4b_run_from_bin.py
python src/template/models/gemma4_e4b/gemma4_e4b_run_from_bin.py --prompt "What is 2+2?"

# Encoder modes require an encoder-containing bin and the runtime gate.
GEMMA4_E4B_ALLOW_ENCODER=1 GEMMA4_PENALTY=1 \
  python src/template/models/gemma4_e4b/gemma4_e4b_run_from_bin.py --vision-enable
GEMMA4_E4B_ALLOW_ENCODER=1 GEMMA4_PENALTY=1 \
  python src/template/models/gemma4_e4b/gemma4_e4b_run_from_bin.py --audio-enable
```

### Deploy footprint

```
gemma4_e4b_run_from_bin.py
gemma4_e4b_config.json
user_dma_core.py + quant_lib.py
gemma4_e4b_bin/
  gemma4_instruction.bin + .json       (15.02 MiB full multimodal)
  weights_gemma4_e4b_hf.bin  + .json   (~10 GB; LM + vision + audio + host sections)
  tokenizer/                           (~32 MB)
```

## Flags

- `--dev xdma0 --cycle 5.62` — DMA device and clock cycle (defaults).
- `--local-weights` — use the local full-model weights bin.
- `--image PATH` implies `--vision-enable`; `--vision-enable` alone
  uses the default image at `../../test_samples/yosemite.jpg`.
- `--audio PATH` implies `--audio-enable`; `--audio-enable` alone
  uses the default audio at `../../test_samples/apex.wav`.
- `GEMMA4_E4B_ALLOW_ENCODER=1` gates VLM/audio runs.
- `GEMMA4_LM_ONLY_BIN=0` during build includes vision/audio instruction
  sections; the default `GEMMA4_LM_ONLY_BIN=1` builds LM-only.
- `GEMMA4_PENALTY=1` enables the on-FPGA repetition penalty used for
  encoder-mode demos.

## Prompt-length limits (dynamic PBI)

The model ships as a single prefill + single decoder program
(`prefill_max_seq_len` and bucket-dispatched decoder; no per-bucket
ISA copies):

- **`prefill_max_seq_len` (config, default `320`)** — prompts up to
  this length are supported. Shorter prompts are padded at the host
  with the last real token; causal bias masks padded positions.
- **Decoder:** single captured program serves all context lengths up
  to `max_context_size` (default `1024`) via the bucket dispatcher in
  `decoder_group_attention_core` (one body per `UE_VECTOR_SIZE`
  multiple of K context length).

Both knobs live in `gemma4_e4b_config.json`.
