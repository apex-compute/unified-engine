# Gemma4 E2B example

Gemma4 E2B accelerator inference. Three modes — text only (LM), image +
text (VLM), and audio + text — run from a single instruction bin.

## Controller placement and current validation

```bash
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma1 --multi-core 2 \
  --prefill-kernel matmatmul --max-new-tokens 32 --prompt "x+3=5, what is x?"
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 --multi-core 8 \
  --prefill-kernel matmatmul --max-new-tokens 32 --prompt "x+3=5, what is x?"
python tests/model_controller_benchmark.py --dev xdma0 --engines 8 --models e2b \
  --prompt "x+3=5, what is x?" --max-new-tokens 32 \
  --json e2b-controller-results.json
```

The existing eight-engine tiled/shared-pool map is preserved. Smaller
configurations now select private windows by the board's memory-controller map
and reserve the shared model span. The two-engine, 4 GiB Kintex image instead
uses two 2 GiB tiles at `0x00000000` and `0x80000000`, one per DDR3 controller.
MLP shards are loaded directly from the host weight file, with no duplicate
shared MLP image. Each tile reserves 1 GiB for private weights, 16 MiB for ISA,
and 64 MiB for private scratch. Compact shared weights (about 540 MiB) and a
contiguous 480 MiB tensor arena fit in the remaining tails. A conservative
703 MiB/core private-weight budget includes both decode and prefill down
layouts and is checked before loading weights. Other multicore configurations
require at least 8 GiB. Program sections in `programs.bin`/`programs.json`
record a placement hash, so equal engine counts on U50 and U55C cannot reuse
instructions with different baked addresses.

The 32-token comparison now passes with exact, meaningful outputs on both
boards for the prompt above:

| Board | Engines | Single-engine decode | Multicore decode | Speedup |
|---|---:|---:|---:|---:|
| Kintex7, 4 GiB | 2 | 224.09 ms/token | 125.80 ms/token | 1.78× |
| Alveo U50, 8 GiB | 8 | 141.18 ms/token | 33.79 ms/token | 4.18× |

The earlier repeatability failure came from the prefill entry: it followed a
32-byte flag-clear instruction, but the absolute-jump encoder rounded its
target up to a 64-byte boundary and skipped the first input DMA load. Prefill
and decoder entries now have explicit 64-byte alignment, runtime dispatch
rejects misaligned targets, and program format version 2 rejects cached images
with the old entry. Fresh Kintex two-engine and U50 eight-engine repeats both
produce the same tokens.
See `../../tests/kintex7_gemma4_e2b_controller_results.json` and
`../../tests/alveo_u50_gemma4_e2b_controller_results.json` for the corrected results.
The older failed comparisons remain diagnostic records of that resolved bug.

`--max-new-tokens` limits generation without changing the compiled context/KV
layout. The comparison runner rejects empty decoded text and padding-only
output, even when token lists match. U55C placement has offline coverage;
U55C hardware results are not available.

## Build and run

Run these commands from the repository root with FPGA access through XDMA.
The Python environment needs PyTorch, Transformers with Gemma4 support,
Hugging Face Hub, and the dependencies for the selected image or audio mode.
Weight generation downloads `google/gemma-4-E2B-it` when the configured local
checkpoint is absent; access to that checkpoint must be available.

`gemma4_e2b_test.py` handles weight preparation, instruction compilation, and
execution. It generates missing `params.bin` weights, then recompiles the
instruction image on each run by default. `--bin-reuse` reuses compatible
cached sections; incompatible or missing sections are compiled again. Omit
`--bin-reuse` to force a fresh instruction build without regenerating weights.

```bash
# Text; --dev selects the board and defaults to xdma0.
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 \
  --prompt "What is 2+2?" --max-new-tokens 32

# Reuse a compatible compiled image.
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 \
  --prompt "What is 2+2?" --max-new-tokens 32 --bin-reuse

# Image or audio input; supply an existing local file.
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 \
  --image my.jpg --prompt "Describe this image." --max-new-tokens 32
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 \
  --audio my.wav --prompt "Describe this audio." --max-new-tokens 32

# Per-phase hardware profiling.
python models/gemma4_e2b/gemma4_e2b_test.py --dev xdma0 --profile
```

The controller comparisons above validate text inference. Image and audio
examples use the same entrypoint and select their respective encoder paths.
`--image` or `--audio` without a filename selects the bundled default sample;
the two modes are mutually exclusive. `--vision-host` with `--image` runs the
vision encoder on the host while keeping LM prefill and decode on the FPGA.

## Files and caches

Paths below are relative to `models/gemma4_e2b/`:

- `gemma4_e2b_test.py`: main CLI and shared engine configuration.
- `gemma4_e2b_lm.py`, `gemma4_e2b_vision.py`, `gemma4_e2b_audio.py`: model stages.
- `gemma4_e2b_config.json`: model dimensions, memory layout, and checkpoint paths.
- `gemma4_e2b_bin/params.bin` and `params.json`: generated weights and manifest.
- `gemma4_e2b_bin/programs.bin` and `programs.json`: captured instruction
  sections and metadata for the selected run configuration.
- `gemma4_e2b_bin/programs_profile.bin` and `programs_profile.json`: separate
  instruction cache for `--profile`.
- `gemma4_e2b_bin/gemma-4-E2B-it/`: local Hugging Face checkpoint.
- `gemma4_e2b_bin/tokenizer/`: bundled tokenizer and processor files.

Cache metadata includes placement, engine count, kernel selection, prompt
length, and program format. Reuse does not make this an execute-only deployment
script; the same CLI still prepares the model and compiles missing sections.

## Engine options and context limits

Clock frequency, AXI width, memory capacity, and available engine count come
from the selected FPGA's `HW_INFO` register. `--multi-core N` selects the number
of engines; bare `--multi-core` selects two. Multicore runs select the
`matmatmul` prefill kernel. `--prefill-kernel`, `--decode-kernel`, and
`--vision-kernel` expose the supported `streaming` or `matmatmul` choices,
subject to hardware validation. Use `--help` for all options.

The checked-in configuration sets `max_prefill_seq_len` and
`prefill_max_seq_len` to 512 and `max_context_size` to 4096. Prefill processes
all prompt tokens except the final one, which starts decoding. Its runtime
row counts follow the actual prompt; the preparation projection uses the
configured template maximum. A dynamic decoder serves subsequent context
lengths. `--max-new-tokens` caps generation without resizing that compiled
context layout. Changes to these configuration limits require recompilation
and sufficient tensor/KV memory.
