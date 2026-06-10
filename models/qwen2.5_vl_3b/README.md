# Qwen2.5-VL-3B

Vision-Language Model (VLM) inference on the FPGA accelerator. Supports image+text
and text-only prompts.

## Layout

- **qwen2.5_vl_3b_test.py** – Full VLM pipeline: vision encoder + prefill + decode
- **qwen2.5_vl_3b_config.json** – Model dimensions, precision, and paths
- **qwen2.5_vl_3b_bin/** – Weights, HF model, and the compiled instruction bin (generated at runtime)

## Architecture

- **LM:** 36 layers, hidden 2048, GQA 16 Q / 2 KV heads (group 8), head_dim 128,
  SwiGLU (intermediate 11008), Q/K/V bias, no QK-norm, mRoPE `[16,24,24]`,
  tied LM head, vocab 151936. IF4 weights (V/O kept BF16).
- **Vision encoder:** 32 layers, hidden 1280, 16 heads, head_dim 80 (padded to
  128 on-chip), SwiGLU (intermediate 3420), RMSNorm, window attention
  (112-px windows) with full attention at layers 7/15/23/31; 2×2 patch merger
  → 144 tokens × 2048. IF4 weights.
- **Compute graphs (mermaid):** see
  `output/compute_graphs/qwen2.5_vl_3b/` —
  `..._vision_encoder_semantic.md` and `..._lm_semantic.md`.

## Performance

Measured on FPGA (kintex7 profile), greedy decode:

| Path | Decode speed | Instruction bin |
| :--- | :--- | :--- |
| Qwen2.5-VL-3B | 2.34 tok/s avg (2.39 peak) | 6.73 MiB (7,060,096 B) — prefill 0.73 + decoder 0.85 + vision encoder 5.15 |

The LM instruction footprint (prefill + decoder = **1.58 MiB**) is fully
dynamic-PBI + §7 shared-subroutine-attention compact. The vision encoder was
**51.65 → 5.15 MiB** (10×) after the Tier-1 shrink (strided-DMA head/token
transposes + combined RoPE + splitting the unrolled norm2/ln_q into M-looped ops;
see the roadmap below).

## Usage

```bash
# VLM with the default image + prompt
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py

# VLM with a custom image + prompt
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image /path/to/photo.jpg --prompt "What do you see?"

# Text-only (the vision encoder is compiled into the bin but not executed)
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image none --prompt "What is 2+2?"
```

### Decode options (on-FPGA repetition penalty)

The repetition penalty is folded into the LM-head matmul bias, so the hardware
argmax returns the penalized token directly — **no logit readback, fully
deterministic**. It is the default; `--pure-greedy` disables it.

```bash
--pure-greedy          # plain greedy decode (zero bias)
--greedy-until N       # pure greedy for the first N decoded tokens, then penalty (default 512)
--pen-alpha A          # bias[t] = -A * count[t] in logit units (default 1.0)
--pen-cap C            # max |bias| per token (default 20.0)
--rep-window W         # count tokens over the last W; punctuation/whitespace/specials exempt (default 256)
```

## Generated files (`qwen2.5_vl_3b_bin/`)

Auto-generated on first run, reused afterward:

- **Weight binaries** (`qwen2.5_vl_3b_lm_if4.bin`, `qwen2.5_vl_3b_vision_if4.bin`) –
  IF4-quantized LM and vision weights, generated once from the HF model.
- **instructions.bin** – The single, mode-agnostic instruction bin: prefill +
  decoder + vision encoder in one file (dynamic PBI; seq-length-agnostic up to
  `prefill_max_seq_len`). Built once on the first run — even a text-only run
  builds the complete bin (canonical vision layout), so the same bin then serves
  both LM-only and VLM requests. LM-only loads it and skips the encoder; VLM runs
  the encoder too.

To force a clean rebuild, delete the cached files:
```bash
rm models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin/instructions.bin
# or, to also regenerate weights:
rm models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin/*.bin
```

## Vision encoder bin-size — optimization roadmap

**Tier 1 — DONE (51.65 → 5.15 MiB, 10×).** All instances of the same principle —
profile per-phase, then convert every per-head/per-row Python-unroll into a
register-driven single op (the LM's pattern):

1. **Head↔token permutes** (was 91% of the layer): `bf16_permute_core_v2` did one
   DMA per row (≈9216 rows × 3–4 permutes/layer). Replaced with `_vis_transpose_10`,
   one strided-DMA gather per output block (16 or 576 iters vs 9216).
2. **RoPE**: 32 per-head calls/layer → 2 M-looped `rope_hf_core_dram(M=VN·VS)`
   calls (cos/sin tiled VN×).
3. **norm2** (the biggest single win, ~2.35 MB): `rms_norm_core_dram_post_add` was
   kept legacy/static and M-unrolled over VS=576 rows (~2400 instr/layer). Split
   into a chunked eltwise-add + an M-looped rms_norm.
4. **merger ln_q**: was missing `gpr_M_reg` (M-unrolled); added it.

Per layer 1578 → ~108 KB; numerics unchanged (compare layer-0 cos=1.0; VLM caption
correct). Total bin **53.23 → 6.73 MiB**.

**Remaining (needs engine work — left as-is):**
- **§7 flash-marshal** (~54 KB/layer, ~1.7 MB): the 144 per-(head,window) marshal
  stubs are loop-shareable in principle, but the loop body `jump`s into the
  register-hungry flash subroutine, which would clobber the loop's counter/offset
  registers (the 15-GPR collision in `shared_design_notes.md` Trick 8). Needs a
  leaner flash register footprint or register-base Q/K/V/out operands in
  `flash_attention_core`.
- **merge transpose** (~37 KB/layer, ~1.2 MB): the gather path loops 576×; a
  strided-**scatter** (16×) gave wrong output on device (verified — a multi-chunk
  strided *write* limitation; the symmetric strided *read* is correct).
- The remaining matmuls (per-layer + merger) are M-looped already; their N-tiling
  can't use PBI because they carry a bias (`shared_design_notes.md` PBI caveats).

**Tier 2 — whole-layer-body sharing (likely unnecessary).** The LM never shares
layer bodies — it unrolls all 36 layers, and is small because each body is small
(M-looped PBI calls). Tier 1 brings the vision body down the same way, so Tier 2
(compile one layer subroutine, loop 32× via a `gpr_weight_base` engine feature)
is probably not needed unless an even smaller bin is required.
