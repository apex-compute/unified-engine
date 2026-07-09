# SmolVLM2

SmolVLM2-500M vision-language inference on the accelerator. **Fixed precision: bf16 vision + q4_64
language.** SigLIP vision encoder (12 layers) → pixel-shuffle connector → SmolLM2 LM (32 layers).

## Layout

- **smolvlm2_test.py** – builds params + compiles the single program bin + runs (prefill + decode).
- **smolvlm2_run_from_bin.py** – runtime-only: loads the bins and runs, no compilation (hard-errors if a
  required bin is missing or its metadata does not match the requested mode).
- **smolvlm2_config.json** – model dims, fixed ISA regs, DRAM layout.
- **smolvlm2_headless_check.py** – stubs the FPGA DMA and runs `compile_all` for a no-hardware
  compile/size check.
- **smolvlm2_bin/** – generated at runtime: the HF model dir (sole weight source) + the two deployable
  bins: **`programs.bin`** + **`programs.json`** (the unified programs bin: encoder + decoder + prefill
  via `compile_all`, with a name->{offset,size,addr} manifest) and **`params.bin`** + **`params.json`**
  (the assembled params snapshot — q4 LM + bf16 vision).

See `../../notes/notes_smolvlm2.md` for the optimization mechanisms (unified attention, strided-DMA
permute elimination, single bin, encoder layer-body sharing, prefill GEMM kernel).

## Prerequisites

- `user_dma_core` is on the path automatically (the scripts self-insert `src/template`), so they run
  from any working directory.
- Python with `torch`, `transformers`, and DMA device access. The first run downloads the HF model.

## Modes (the only flags beyond device/prompt/image)

| Flag | Effect |
|------|--------|
| *(none)* | **Default.** Decode linear layers use the fused `quantized_matmat_core` kernel; generation uses the **on-FPGA repetition penalty** (the LM-head's `PENALTY_BIAS_DRAM` C term) to keep long outputs from looping. The penalty is a **runtime tensor write only**, so it never changes the compiled bin. |
| `--greedy-enable` | Use **pure greedy** decoding (penalty off). Same bin; only the runtime bias tensor differs (zeros = greedy). |
| `--decode-matmat_mul_core-enable` | **Deterministic control path.** Every *per-layer* decoder and prefill linear (Q/K/V/O/gate/up/down) runs through the general IF4 `matmat_mul_core(is_B_quantized=True)` path instead of the fused dot-product kernel (which can show rare run-to-run ULP differences on bit-identical inputs). Same packed IF4 weights + scales; only the kernel differs. The final LM-head GEMV stays on the fused `quantized_matmat_core` in both modes (the optimized GEMV for the 49280-wide vocab + on-chip argmax). |

The default run is **VLM** (vision): with no `--image` it uses the bundled sample image
(`test_samples/vette.jpg`) and the prompt *"Describe this image."* Pass `--image PATH` for a different
image, or `--lm-enable` (or `--image none`) for **pure LM** text-only mode (skips the vision encoder).
The program bin is **always the full VLM bin** (encoder + decoder + prefill), built once and loaded;
the mode only decides whether the encoder runs.

## Artifacts (`smolvlm2_bin/`)

The artifact name carries **only** the decode-kernel suffix (the penalty is runtime-only and never
suffixes the name):

| Mode | Params | Programs |
|------|---------|--------------|
| default (fused) | `params.bin` / `params.json` | `programs.bin` / `programs.json` |
| `--decode-matmat_mul_core-enable` | `params_decode_matmat_mul_core.bin` / `.json` | `programs_decode_matmat_mul_core.bin` / `.json` |

The two `params*.bin` files are byte-identical (the decode kernel changes only the program stream,
not the params layout); they are kept per-mode so `run_from_bin` can validate mode metadata uniformly.

## Manual commands

`smolvlm2_test.py` **builds on the first run** for a given mode (then loads the cached bin); pass the
same flags to `smolvlm2_run_from_bin.py` to **load and run** the bin that `test.py` produced — it never
compiles.

```bash
# ---- Default: VLM (fused decode + on-FPGA repetition penalty), bundled sample image --------
# builds params.bin + programs.bin on first run
python models/smolvlm2/smolvlm2_test.py
# run from the bin (no compile, fully offline)
python models/smolvlm2/smolvlm2_run_from_bin.py
# VLM with your own image / prompt
python models/smolvlm2/smolvlm2_run_from_bin.py --image test_samples/vette.jpg --prompt "Describe this image."

# ---- Pure LM (text-only) ------------------------------------------------------------------
python models/smolvlm2/smolvlm2_test.py --lm-enable --prompt "What is the capital of France?"
python models/smolvlm2/smolvlm2_run_from_bin.py --lm-enable --prompt "What is the capital of France?"

# ---- Pure greedy (penalty off; same bin) --------------------------------------------------
python models/smolvlm2/smolvlm2_run_from_bin.py --greedy-enable --prompt "What is the capital of France?"

# ---- Deterministic control path (per-layer IF4 matmat_mul_core; LM-head stays fused) -------
# Builds params_decode_matmat_mul_core.bin + programs_decode_matmat_mul_core.bin on first run
python models/smolvlm2/smolvlm2_test.py --decode-matmat_mul_core-enable \
    --prompt "What is the capital of France?"
python models/smolvlm2/smolvlm2_run_from_bin.py --decode-matmat_mul_core-enable \
    --image test_samples/vette.jpg --prompt "Describe this image."
```
