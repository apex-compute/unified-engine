# LLaMA 3.2 3B

LLaMA 3.2 3B prefill + decode on the accelerator. Same architecture family as 1B,
but 28 layers, hidden=3072, 8 KV heads, group_size=3, **per-head RoPE (N=128)**.
4096-token context, dynamic-PBI single instruction bin, sampling decode.

## Files

| File | Purpose |
|---|---|
| `llama3.2_3b_test.py`         | First run: downloads the HF model, generates the weight bin + unified instruction bin, then runs. Later runs reuse the cached bins. |
| `llama3.2_3b_run_from_bin.py` | Runtime-only: loads the pre-built bins and runs. No HuggingFace access, no recompile, no import from the test script. Fails loudly if a required artifact is missing. |
| `llama3.2_3b_config.json`     | Model dims + weight-bin region offsets. |
| `llama3.2_3b_bin/`            | Generated artifacts (created on first run). |

## Usage

Run from the repo root (so `user_dma_core` is on the path), or from this folder.

```bash
# Default prompt
python llama3.2_3b_test.py
python llama3.2_3b_run_from_bin.py

# Custom prompt (uses the model's chat template)
python llama3.2_3b_test.py --prompt "What is 2+2?"

# Sampling controls (defaults: temperature 0.6, top-p 0.9, repetition-penalty 1.2)
python llama3.2_3b_test.py --prompt "..." --temperature 0.6 --top-p 0.9 --seed 7
python llama3.2_3b_test.py --prompt "..." --temperature 0        # greedy (HW argmax)

# DMA device / clock period
python llama3.2_3b_test.py --dev xdma0 --cycle 5.88
```

Both scripts share `--prompt`, `--dev`, `--cycle`, and the sampling flags
`--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--rep-window`,
`--rep-decay`, `--seed`. Prompts up to 128 tokens are supported (`prefill_context_size`).

## First run vs cached run

| Step | First `test.py` run | Cached `test.py` / any `run_from_bin.py` |
|---|---|---|
| HF model download | once (cached after) | skipped |
| Build weight bin (quantize) | ~few min | skipped |
| Compile unified instruction bin (~63 MB) | ~8 s | skipped — loaded from disk |
| Prefill + decode | runs | runs |

## Forcing a rebuild

```bash
# Recompile the instruction bin (e.g. after editing the compile path)
rm llama3.2_3b_bin/programs.bin llama3.2_3b_bin/programs.json
# (and llama3.2_3b_bin/programs_fpgapenalty.{bin,json} for the on-FPGA penalty build)

# Re-quantize the weight bin (from the cached HF model)
rm llama3.2_3b_bin/params.bin llama3.2_3b_bin/params.json
```

## Shipping `run_from_bin`

To run on a machine with no HuggingFace access / no internet, ship
`llama3.2_3b_run_from_bin.py`, `llama3.2_3b_config.json`, and the
`llama3.2_3b_bin/` artifacts — the weight bin, the instruction bin + meta, and the
tokenizer files under `Llama-3.2-3B-Instruct/` (`tokenizer.json` /
`tokenizer_config.json`; the model safetensors are **not** needed). `run_from_bin.py`
imports nothing from the test script and never contacts HuggingFace.

## DRAM layout (4096 context)

Full-4 GB layout set in the engine `__init__` (params is bigger than 1B, so the
tensor base moves up vs 1B):

| Region | Base | Size | Used @ 4096 |
|---|---|---|---|
| Params  | `0x00000000` | 1.75 GiB | ~1.61 GiB (weights + RoPE) |
| Tensor  | `0x70000000` | 1.75 GiB | ~1.59 GiB (KV cache + activations) |
| Program | `0xE0000000` | 512 MiB  | ~63 MB (instruction bin) |

Total ~3.3 GB of 4 GB. (The HF embedding stays host-side.)
