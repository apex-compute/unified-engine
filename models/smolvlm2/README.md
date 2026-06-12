# SmolVLM2

SmolVLM2-500M vision-language inference on the accelerator. **Fixed precision: bf16 vision + q4_64
language.** SigLIP vision encoder (12 layers) → pixel-shuffle connector → SmolLM2 LM (32 layers).

## Layout

- **smolvlm2_test.py** – builds weights + compiles + runs (prefill + decode).
- **smolvlm2_run_from_bin.py** – runtime-only: loads the bins and runs, no compilation.
- **smolvlm2_config.json** – model dims, fixed ISA regs, DRAM layout.
- **smolvlm2_headless_check.py** – stubs the FPGA DMA and runs `compile_all` for a no-hardware
  compile/size check.
- **smolvlm2_bin/** – generated at runtime: the HF model dir (sole weight source) + the two deployable
  bins: **`instructions.bin`** (the one instruction bin: encoder + decoder + prefill via `compile_all`)
  and **`weights.bin`** (the assembled params snapshot — q4 LM + bf16 vision).

See `../../notes/notes_smolvlm2.md` for the optimization mechanisms (§7 shared attention, strided-DMA
permute elimination, single bin, encoder layer-body sharing, prefill GEMM kernel).

## Prerequisites

- Run from the **repo root directory** so `user_dma_core` is on the path.
- Python with `torch`, `transformers`, and DMA device access. First run downloads the HF model.

## Usage

The instruction bin is **always the full VLM bin** (encoder + decoder + prefill) — built once on the
first run, then loaded (cache keyed on the layout signature). **Vision is opt-in at runtime:** the
default run is **LM-only** (text); pass `--image` (or `--vision-enable`) to run the vision encoder.

```bash
# LM-only (text) — the default, no vision encoder runs
python models/smolvlm2/smolvlm2_test.py --prompt "What is the capital of France?"

# VLM — pass an image (or --vision-enable to use the default sample image)
python models/smolvlm2/smolvlm2_test.py --image test_samples/vette.jpg --prompt "Describe this image."
python models/smolvlm2/smolvlm2_test.py --vision-enable

# Runtime-only (load the pre-built bins, no compilation). Same LM-only default / --image opt-in.
python models/smolvlm2/smolvlm2_run_from_bin.py
python models/smolvlm2/smolvlm2_run_from_bin.py --image test_samples/vette.jpg
```
