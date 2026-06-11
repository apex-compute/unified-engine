# SmolVLM2

SmolVLM2-500M vision-language inference on the accelerator. **Fixed precision: bf16 vision + q4_64
language.** SigLIP vision encoder (12 layers) → pixel-shuffle connector → SmolLM2 LM (32 layers).

## Layout

- **smolvlm2_test.py** – builds weights + compiles + runs (prefill + decode).
- **smolvlm2_run_from_bin.py** – runtime-only: loads the bins and runs, no compilation.
- **smolvlm2_config.json** – model dims, fixed ISA regs, DRAM layout.
- **smolvlm2_bin/** – generated at runtime: the HF model dir (sole weight source) + the two deployable
  bins: **`instructions.bin`** (the one instruction bin: encoder + decoder + prefill via `compile_all`)
  and **`weights.bin`** (the assembled params snapshot — q4 LM + bf16 vision).

See `../../notes/notes_smolvlm2.md` for the optimization mechanisms (§7 shared attention, strided-DMA
permute elimination, single bin, encoder layer-body sharing, prefill GEMM kernel).

## Prerequisites

- Run from the **repo root directory** so `user_dma_core` is on the path.
- Python with `torch`, `transformers`, and DMA device access. First run downloads the HF model.

## Usage

```bash
# Prefill + decode (default prompt + image). First run builds weights.bin + instructions.bin;
# subsequent runs load them automatically (cache keyed on the layout signature).
python models/smolvlm2/smolvlm2_test.py

# Custom prompt and image
python models/smolvlm2/smolvlm2_test.py --prompt "Describe this image." --image test_samples/vette.jpg

# Runtime-only (load the pre-built bins, no compilation)
python models/smolvlm2/smolvlm2_run_from_bin.py
```
