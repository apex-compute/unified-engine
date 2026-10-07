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
| `qwen3.5_2b_bin/programs.bin` (+ `programs.json`) | Unified programs bin (**4.32 MB** = encoder 0.82 + decoder 3.50). Manifest lists `{programs: {encoder: {offset, size}, decoder: {offset, size}}}`. |
| `qwen3.5_2b_bin/params.bin` (+ `params.json`) | Quantized weights (built on first run) + manifest. |

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

## Controller-private decode

The builder supports two-engine text generation on a 4 GiB Kintex7 and larger
engine counts on supported U50/U55C images:

```bash
python models/qwen3.5_2b/qwen3.5_2b_test.py --device kintex7 --dev xdma1 \
  --multi-core 2 --max-new-tokens 128 --prompt "What is 2 + 2?"
python models/qwen3.5_2b/qwen3.5_2b_test.py --device alveo --dev xdma0 \
  --multi-core 8 --max-new-tokens 128 --prompt "What is 2 + 2?"
```

Each engine reads its own column slices of the IF4 projections and LM head.
The fused full-attention KV projection stays fused; Gated DeltaNet recurrence,
convolution state, attention, normalization, and residual operations remain on
the primary engine. Every prompt token and generated token launches matching
worker programs, and the LM head merges the per-engine argmax results.

Kintex7 uses these nonoverlapping regions:

| Data | Address range | Observed allocation at context 512 |
|---|---|---|
| Engine 0 private weights/ISA/scratch | 0–512 MiB, DDR0 | 476.27 MiB weights |
| Shared parameters | 1–2 GiB | 964.00 MiB |
| Shared recurrent state and activations | 2–2.5 GiB | 130.28 MiB persistent; transient activations reuse the remainder |
| Primary instructions | 2.5–3 GiB | 7.64 MiB decoder |
| Engine 1 private weights/ISA/scratch | 3–3.5 GiB, DDR1 | 476.27 MiB weights, 215 KiB worker instructions |

Each private window reserves 16 MiB for instructions and 16 MiB for scratch,
leaving a 480 MiB weight budget. The complete shard plan is checked before DMA;
this model has only 3.73 MiB spare per engine in the two-engine layout. The host
keeps the BF16 embedding table. The original single-engine map stays unchanged.
U50 follows the hardware SAXI controller order, with the shared model at 6–8 GiB.

Multicore currently supports text generation through the builder. It compiles
primary and worker instructions together after every weight load; legacy
`programs.bin` files remain single-engine artifacts. The runtime-only loader
rejects multicore artifacts because it cannot restore the workers. Transient
single-engine decoder caches include source, context-capacity, and memory-map
identity. Identity-matrix addresses are scoped to each engine instance so that
switching between single-engine and multicore layouts in one process is safe.

The eleven offline tests in `tests/test_qwen35_multicore.py` cover placement,
capacity, recurrent prefill/decode orchestration, fused projection registration,
cache rejection, and worker startup. A full capture with the real quantized
weights also passed without device access.

Both boards generated 32 identical token IDs between the single-engine baseline
and multicore run for `x+3=5, what is x?`. Average hardware latency over the
31 timed decode steps was:

| Board | Engines | Single engine | Multicore | Speedup | Result |
|---|---:|---:|---:|---:|---|
| Kintex7, image `0x8763d976`, 198.324 MHz | 2 | 1271.14 ms/token | 1190.33 ms/token | 1.068× | [JSON](../../kintex7_qwen35_controller_results.json) |
| U50, image `0xe6703022`, 333.332 MHz | 8 | 776.70 ms/token | 689.38 ms/token | 1.127× | [JSON](../../alveo_u50_qwen35_controller_results.json) |

The prompt seed token is included in the token comparison and excluded from
the timing average.
The 18 recurrent-attention layers remain on the primary engine, limiting the
end-to-end gain even when projection weights use separate memory controllers.
This measurement does not establish a 12 GB/s whole-model decoding rate.

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

LM-only **execution** skips the encoder (it is still *built* into the comprehensive
bin on the first run — see Load-only below — just not run).

The LM head runs **on the FPGA**: after the final norm, a quantized (FP4) matmul of
the tied embedding produces logits, the per-vocab penalty bias is added as its C
term, and the hardware argmax returns the next token (no logits read back).

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

## Sampling

Decode is **pure greedy by default**. The quantized FP4 LM head and hardware
argmax return the next token without reading logits back to the host.

An optional on-FPGA repetition penalty can be enabled with
`Q35_REPETITION_PENALTY=1`. It uses a per-vocabulary bias as the LM-head C term
(`bias_mode="broadcast_N"`) and rebuilds the bias each step as
`bias[t] = clamp(-α·count[t], min=-cap)` over a window of recent tokens;
punctuation/whitespace/special tokens are exempt, and a token repeated
`PEN_LOOP_RUN` times in a row is hard-banned to break stuck loops.

## Environment toggles (all optional)

| Var | Effect |
|-----|--------|
| `Q35_REPETITION_PENALTY=1` | Enable the optional on-FPGA repetition penalty. |
| `Q35_PURE_GREEDY=1` | Force pure greedy even if the penalty was enabled. |
| `PEN_ALPHA` | Penalty strength `bias=-α·count` (default `3.0`). |
| `PEN_CAP` | Penalty floor `clamp(min=-cap)` (default `20.0`). |
| `REP_WINDOW` | Token-frequency window, last N decoded (default `256`). |
| `GREEDY_UNTIL` | Pure greedy for the first N decoded tokens (default `8`). |
| `PEN_LOOP_RUN` | ≥N identical tokens in a row ⇒ hard ban (default `4`). |
| `VIS_LEGACY=1` | Use the unrolled multi-capture vision encoder instead of the compact §7/§3b one. |
| `Q35_NO_S7_FLASH=1` | Disable the shared §7 flash subroutine (inline bodies — bigger bin). |
| `Q35_VIS_LN_STATIC=1` | Static (non-PBI) vision LayerNorm fallback. |

See `../../notes/notes_qwen3.5_2b.md` for the full implementation notes
(bin minimization, vision compaction, and the single-bin design + root cause).
