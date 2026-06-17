# Qwen3.5-2B on FPGA — LM + VLM in a single instruction bin

Hybrid LLM (18 Gated-DeltaNet linear-attention layers + 6 full-attention layers)
+ a 24-layer ViT vision encoder, quantized to **FP4_64**, running on the unified
FPGA engine.  Everything — vision encoder **and** LM decoder — lives in ONE
instruction bin.

## Files

| File | Role |
|------|------|
| `qwen3.5_2b_test.py` | Builder + reference runner. Compiles the model, **generates the unified bin**, and runs LM/VLM inference. |
| `qwen3.5_2b_run_from_bin.py` | Customer-facing **runtime-only** runner. Loads the prebuilt bin and runs — no compilation. |
| `qwen3.5_2b_config.json` | Architecture + paths config. |
| `qwen3.5_2b_bin/qwen3.5_2b_instruction.bin` (+ `.json`) | The unified instruction bin (**4.32 MB** = encoder 0.82 + decoder 3.50) + meta. |
| `qwen3.5_2b_bin/weights_qwen3.5_2b_hf.bin` | Quantized weights (built on first run). |

## Quick start

```bash
# 1) First run BUILDS the comprehensive unified bin (encoder + decoder).
#    Either mode works — both write the same full bin:
python3 qwen3.5_2b_test.py --vision-enable --vision-on-hardware   # VLM: builds + captions
python3 qwen3.5_2b_test.py --prompt "Tell me about the Eiffel Tower."  # LM-only: still builds the full bin

# 2) LM-only text generation (test.py or the runtime-only runner):
python3 qwen3.5_2b_test.py     --prompt "Tell me about the Eiffel Tower."
python3 qwen3.5_2b_run_from_bin.py --prompt "Tell me about the Eiffel Tower."

# 3) VLM image caption from the prebuilt bin:
python3 qwen3.5_2b_run_from_bin.py --vision-enable
python3 qwen3.5_2b_run_from_bin.py --image my.jpg --prompt "What is in this image?"
```

`--vision-enable` uses the bundled sample image (`../../test_samples/yosemite.jpg`).
Decode runs at ~1.1 tok/s (greedy).

## How the single bin works

The file holds two program sections, **both baked for program base 0xD0000000**,
run **sequentially with a DRAM reset between them** (they are never resident
together):

```
[encoder section]  vision encoder — vision weights + program at base
[decoder section]  LM decoder     — LM weights     + program at base
```

VLM execution order (this order is mandatory — see below):

```
run vision encoder → merged image tokens to host
reset DRAM (params/tensor/program bump pointers)
prepare_inference(LM)              # exactly ONCE
load decoder section @ base
prefill (replay decoder per prompt token) + decode
```

LM-only skips the encoder entirely.

**Load-only:** the **first** run of `test.py` — LM-only *or* VLM — builds the
encoder + decoder and writes the comprehensive bin (an LM-only first run still
builds the encoder section, using the bundled sample image, so the single bin
always holds the full model; it just isn't executed). **Every later run** of
`test.py` AND all runs of `run_from_bin` LOAD both program sections from the bin —
nothing is recompiled.

### ⚠ Two rules baked into this design

1. **`prepare_inference()` exactly once per engine, and vision runs *before* it.**
   A second prepare — or running the vision encoder *after* prepare — silently
   corrupts the LM decode (prints `!!!!…`) and is **not** recoverable by
   `software_reset()`/`clear_dram()`.  The bin runs encoder→reset→prepare→decoder.

2. **The LayerNorm zeros base is a shared, pre-seeded buffer (`VIS_ZEROS`).**
   Kernel primitives like `layer_norm_core_dram` need a constant `zeros` vector in
   DRAM that they read at run time. Rather than let the kernel `dma_write` one per
   call at *compile* time (which a bin-load can't recreate → all-NaN tokens), the
   template reserves one `VIS_ZEROS` buffer, seeds it once in `setup_only`, and passes
   `ZEROS_DRAM_ADDR=VIS_ZEROS` to every LayerNorm — the same mechanism as the identity
   matrix. `setup_only` runs on the load path too, so it "just works." See
   `../../notes/shared_design_notes.md` (Trick 9).

## Environment toggles (all optional)

| Var | Effect |
|-----|--------|
| `VIS_LEGACY=1` | Use the unrolled multi-capture vision encoder instead of the compact §7/§3b one. |
| `Q35_NO_S7_FLASH=1` | Disable the shared §7 flash subroutine (inline bodies — bigger bin). |
| `Q35_VIS_LN_STATIC=1` | Static (non-PBI) vision LayerNorm fallback. |
| `Q35_INSTR_BREAKDOWN=1` | Print a per-op instruction-size breakdown (diagnostic). |

See `../../notes/notes_qwen3.5_2b.md` for the full implementation notes
(bin minimization, vision compaction, and the single-bin design + root cause).
