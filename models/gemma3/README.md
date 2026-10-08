# Gemma3 example

This folder contains the Gemma3 accelerator inference example and numeric verification.

> [!WARNING]
> `gemma3_test_IF8.py` is deprecated and IF8 is currently not working. It is
> retained only for historical/debugging reference, excluded from
> `model_auto_test.py`, and must not be treated as a supported inference path.
> Use `gemma3_test.py` (IF4).

## Layout

- **gemma3_test.py** – Prefill + decode loop on accelerator (single or multi engine, `--multi-core N`).
- **../../multi_engine_shard.py** – the shared multi-engine library. The decoder's
  N-sharded matmuls, batch-split attention and master/worker rendezvous all come
  from here (`MultiEngineScheduler`). `model_multicore_layout()` selects disjoint
  model storage and private weight windows for each supported board.
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

# DMA device; clock, AXI width, memory size, and core count come from HW_INFO
python models/gemma3/gemma3_test.py --dev xdma1

# Use local full-model weights bin
python models/gemma3/gemma3_test.py --local-weights

```

## Decoder multi-core (`--multi-core N`)

The decoder's quantized matmuls are N-sharded (split by output columns) across
up to 12 engines, bounded by the core count reported by `HW_INFO`. Each engine
holds its own column block in a private window selected for that board's memory
controllers. Engines synchronize each sharded round through the four-phase
`FLAG_CHECK_SET` / `FLAG_CHECK_CLEAR` handshake. Prefill uses the original full
weights on engine 0.

```bash
# P2 Kintex-7: two engines on separate DDR controllers
python models/gemma3/gemma3_test.py --dev xdma1 --multi-core 2

# Alveo U50: eight engines using the HBM controller map
python models/gemma3/gemma3_test.py --dev xdma0 --multi-core 8
```

Device numbers depend on the host; the examples above match the tested P2
configuration. Board selection uses the reported hardware signature even when
only two engines of an Alveo board are active.

| Board signature | Shared model range | Private weight windows |
|---|---|---|
| Kintex-7, 2 cores / 4 GiB | `[1, 3)` GiB | 512 MiB at 0 and 3 GiB, on separate DDR controllers |
| U50, 8 cores / 8 GiB | `[6, 8)` GiB | Primary 512 MiB HBM segments in hardware engine/SAXI order |
| U55C, 12 cores / 16 GiB | `[6, 8)` GiB | One 1 GiB controller region per engine; a reserved region moves to the same MC on the other stack when available |
| U55C, 12 cores / 8 GiB | `[6, 8)` GiB | Up to six 1 GiB windows; 7–12 engines use disjoint 512 MiB segments and share some controllers |

The shared model keeps its original internal offsets within a reserved 2 GiB
span: weights at the base, tensors at `+0x30000000`, and instructions at
`+0x50000000`. Allocations and instruction writes are checked against these
bounds before they can overwrite a private shard. Single-engine runs retain the
original addresses.

The raw Kintex memory benchmark uses windows 2 GiB apart. Model inference uses
windows at 0 and 3 GiB so the full shared model fits between them; both choices
place the two private shards on different DDR controllers. The U55C 8 GiB image
cannot provide twelve independent controller regions alongside the model.

Compiled multi-engine filenames include a hash of the model base and ordered
private windows, so a binary from another placement cannot be reused accidentally.
`--bin-reuse` remains available for single-engine runs. Multi-engine runs always
recompile the primary image and its persistent worker programs together.

### Measured validation

Both comparisons used the same 28-token prefill and produced exactly the same
70 decoded token IDs, including the stop token, and decoded text as their
single-engine baseline. The prompt was `Solve 2x + 3 = 7. Reply with only the value
of x.` First-token rates below use hardware execution time.

| Board | Engines | First-token rate, single → multi | First-token speedup | Mean hardware decode time, single → multi |
|---|---:|---:|---:|---:|
| Kintex-7, 198.324 MHz | 1 → 2 | 10.60 → 19.64 tok/s | 1.85× | 95.77 → 52.16 ms/token |
| U50, 333.332 MHz | 1 → 8 | 16.83 → 76.80 tok/s | 4.56× | 60.28 → 13.11 ms/token |

See the [Kintex-7 results](../../tests/kintex7_gemma3_controller_results.json) and
[U50 results](../../tests/alveo_u50_gemma3_controller_results.json) for the complete
hardware signatures, prompt, token IDs, placements, and timing measurements.
These are end-to-end decode measurements, not a claim that every model operation
sustains the raw memory benchmark's bandwidth. U55C placement is covered by
software tests; these results do not include a U55C hardware run.

### What is sharded

Optional shards are controlled by the flags in `gemma3_test.py`:

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
