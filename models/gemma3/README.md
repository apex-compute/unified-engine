# Gemma3 example

This folder contains the Gemma3 accelerator inference example and numeric verification.

> [!WARNING]
> `gemma3_test_IF8.py` is deprecated and IF8 is currently not working. It is
> retained only for historical/debugging reference, excluded from
> `model_auto_test.py`, and must not be treated as a supported inference path.
> Use `gemma3_test.py` (IF4).

## Layout

- **gemma3_test.py** – Prefill + decode loop on accelerator (single or multi engine, `--multi-core N`).
- **multi_engine_shard_gemma3.py** – N-sharded quantized matmul for the decoder, up to 8 engines.
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

## Decoder multi-core (`--multi-core N`)

The decoder's quantized matmuls are N-sharded (split by output columns) across up to 8
engines. Each engine holds a private copy of its own column block in the low 2 GB; the
main PARAMS/TENSOR/PROGRAM map at `0x8000_0000` is untouched. Engines rendezvous once
per sharded round through the four-phase `FLAG_CHECK_SET` / `FLAG_CHECK_CLEAR` handshake.

```bash
python models/gemma3/gemma3_test.py --multi-core 8
```

Measured on hardware (366.7 MHz, IF4, gemma3-1B):

| cores | 1st-token tok/s | speedup | of ideal | avg tok/s | avg GFLOPS | % N-core peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.63 | 1.00x | 100% | 17.82 | 37.02 | 78.9% |
| 2 | 29.21 | 1.57x | 78% | 27.46 | 57.64 | 61.4% |
| 4 | 40.82 | 2.19x | 55% | 37.40 | 79.97 | 42.6% |
| 8 | 82.77 | 4.44x | 56% | 52.53 | 114.03 | 30.4% |

**Read the first-token column, not the average.** First-token speed is measured at the
same short context in every run, so it is the one comparable number. The 8-core row was
captured from a different run (14-token prompt, 498 tokens decoded to a 512 context,
versus 19/76 for the others), and its *average* therefore includes long-context decode
where attention grows and the unsharded work dominates -- it is not comparable with the
rows above it.

`% N-core peak` falls as cores rise because the denominator is the aggregate peak
(`freq x 128 x cores`) while attention, the norms, the residuals and the gate*up multiply
all still run on engine 0 alone. That falling number is Amdahl's law made visible, not a
regression.

### What is sharded

Each op is behind its own flag at the bottom of `gemma3_test.py`:

| op | K | N | 8-way split | flag |
|---|---:|---:|---|---|
| Q proj | 1152 | 1024 | 128 x 8 (even) | `SHARD_QKV` |
| K / V proj | 1152 | 256 | **not splittable past 4 engines** (4 blocks of 64) | `SHARD_QKV` |
| attn O proj | 1024 | 1152 | 128 x 6 + 192 x 2 | `SHARD_ATTN_OPROJ` |
| MLP gate | 1152 | 6912 | 832 x 4 + 896 x 4 | always on |
| MLP up | 1152 | 6912 | 832 x 4 + 896 x 4 | always on |
| MLP down | 6912 | 1152 | 128 x 6 + 192 x 2 | `SHARD_MLP_DOWN` |
| LM head | 1152 | 262144 | 32768 x 8 (even) | always on |

A column shard must be a whole multiple of `UE_VECTOR_SIZE` (64), so remainders are
handed to the trailing engines and engine 0 -- which also runs everything unsharded --
takes the smallest block. Ops that share an input and write disjoint outputs ride in one
round: Q/K/V together, and gate+up together.
