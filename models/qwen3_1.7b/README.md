# Qwen3 1.7B

Qwen3-1.7B inference on the accelerator.

## Files

| File | Purpose |
|---|---|
| `qwen3_1.7b_test.py` | Build + run. Downloads the HF model, generates the weight bin, compiles the instruction bin, then runs prefill + decode. Needs network on first run. |
| `qwen3_1.7b_run_from_bin.py` | Run-only. Loads pre-compiled bins and tokenizer from disk and runs prefill + decode. Fully offline. |
| `qwen3_1.7b_config.json` | Model + DRAM layout config. |
| `qwen3_1.7b_bin/` | Per-run artifacts (weight bin, instruction bin + meta, HF model + tokenizer cache). |

## Prerequisites

- Run from the **repo root** so `user_dma_core` is on `sys.path`.
- Python with `torch`, `transformers`, and DMA access (`/dev/xdma0_*`).
- Between runs that may have left the FPGA in a bad state: `python user_hw_test.py` (software reset).

## Usage

```bash
# First run (with network — downloads HF model, builds weight + instruction bins):
python models/qwen3_1.7b/qwen3_1.7b_test.py --prompt "What is 2+2?"

# Subsequent runs from cached bins (offline-safe):
python models/qwen3_1.7b/qwen3_1.7b_run_from_bin.py --prompt "What is 2+2?"
```

**Prompt-length range:** the cached bin handles any prompt from 1 to
`prefill_max_seq_len` tokens (default **256**) without recompile.
Prefill is a single seq_len-agnostic PBI program; the bucket cap on
flash attention is what sets the upper bound. Bump
`model.prefill_max_seq_len` in the config and delete the cached bin
to extend the range (bin grows roughly linearly with that knob).

Force a recompile of the instruction bin by deleting both files:

```bash
rm qwen3_1.7b_bin/programs.bin qwen3_1.7b_bin/programs.json
```

## CLI flags (both scripts)

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | `default_prompt` from config | User prompt; wrapped with the system prompt by `apply_chat_template`. |
| `--dev` | `xdma0` | DMA device name. |
| `--cycle` | `5.62` | Clock period (ns). Peak GFLOPS = `128 / cycle` ≈ 22.8 at 5.62 ns. |
| `--local-weights` | off | Use `qwen3_1.7b_bin/full_model_weights.bin` instead of the HF-generated bin. |

## Expected output

```
--- Starting decoder ---
<think>
</think>

2 + 2 equals 4.
Stop token 151645 reached.
Decoder done in ~8.4s, total 38 tokens, decode speed: ~2.7 tokens/s.
```

Prefill reports about 19.9 GFLOPS at 5.62 ns clock (~87 % of 22.8 GFLOPS peak).
