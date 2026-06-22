# Qwen3 4B

Qwen3-4B inference on the accelerator.

## Files

| File | Purpose |
|---|---|
| `qwen3_4b_test.py` | Build + run. Downloads the HF model, generates the weight bin, compiles the instruction bin, then runs prefill + decode. Needs network on first run. |
| `qwen3_4b_run_from_bin.py` | Run-only. Loads pre-compiled bins and tokenizer from disk and runs prefill + decode. Fully offline. |
| `qwen3_4b_config.json` | Model + DRAM layout config. |
| `qwen3_4b_bin/` | Per-run artifacts (weight bin, instruction bin + meta, HF model + tokenizer cache). |

## Prerequisites

- Run from the **repo root** so `user_dma_core` is on `sys.path`.
- Python with `torch`, `transformers`, and DMA access (`/dev/xdma0_*`).
- Between runs that may have left the FPGA in a bad state: `python src/template/user_hw_test.py` (software reset).

## Usage

```bash
# First run (with network — downloads HF model, builds weight + instruction bins):
python src/template/models/qwen3_4b/qwen3_4b_test.py --prompt "What is 2+2?"

# Subsequent runs from cached bins (offline-safe):
python src/template/models/qwen3_4b/qwen3_4b_run_from_bin.py --prompt "What is 2+2?"
```

**Prompt-length range:** the cached bin handles any prompt from 1 to
`prefill_max_seq_len` tokens (default **256**) without recompile.
For long prompts (>50 tokens) prefill can take 25-30 s wall-clock because
flash attention's high bucket bodies do `seq_len²` work × 8 KV heads ×
36 layers. The `program_execute` wait_queue timeout is set to 120 s
in this template to absorb that.

Force a recompile of the instruction bin by deleting both files:

```bash
rm qwen3_4b_bin/programs.bin qwen3_4b_bin/programs.json
```

## CLI flags (both scripts)

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | `default_prompt` from config | User prompt; wrapped with the system prompt by `apply_chat_template`. |
| `--dev` | `xdma0` | DMA device name. |
| `--cycle` | `5.62` | Clock period (ns). Peak GFLOPS = `128 / cycle` ≈ 22.8 at 5.62 ns. |
| `--local-weights` | off | Use `qwen3_4b_bin/full_model_weights.bin` instead of the HF-generated bin. |

## Expected output

```
--- Starting decoder ---
<think>
... [reasoning] ...
</think>
The sum of 2 + 2 is 4.
**Answer:** 4.
Stop token 151645 reached.
Decoder done in ~120s, decode speed: ~1.19 tokens/s.
```

Prefill reports about **20.12 GFLOPS** at 5.62 ns clock (~88 % of 22.8 GFLOPS peak).

## Footprint at run time (~3.0 GB / 4 GB)

- Params DRAM: **2.15 GB** (36 layers + LM head + ROPE + per-call identities).
- Tensor DRAM: **730 MB** (KV cache 288 MB + activations 442 MB).
- Program DRAM: **58 MB** (one prefill + one decoder program).
- Host: 778 MB bf16 embedding stays in CPU RAM (per-token row pushed inline).

## More detail

Design and architecture notes: `src/template/notes/notes_qwen3_4b.md`.
Cross-model dynamic-PBI design that these scripts implement: `src/template/core_changes.md`.
