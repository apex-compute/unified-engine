# LLaMA 3.2 1B

LLaMA 3.2 1B prefill + decode on the accelerator.

## Files

| File | Purpose |
|---|---|
| `llama3.2_1b_test.py`         | First-run setup: downloads the HF model, generates weight bin + unified PBI instruction bin, then runs. Subsequent runs reuse the cached bins. |
| `llama3.2_1b_run_from_bin.py` | Runtime-only: loads the pre-built bins and runs. No HuggingFace access, no recompile. Fails loudly with a clear error if any required file is missing. |
| `llama3.2_1b_config.json`     | Model dims, bucket lists, weight-bin region offsets. |
| `llama3.2_1b_bin/`            | Generated artifacts (created on first run). |

After the first successful `llama3.2_1b_test.py` run, `llama3.2_1b_bin/` contains:

```
llama3.2_1b_bin/
├── weights_llama3.2_1b_hf.bin          # IF4 weights + bf16 embedding (~620 MB)
├── llama3.2_1b_instruction.bin         # unified PBI instruction bin (~400 MB)
├── llama3.2_1b_instruction.json        # per-bucket offsets + FLOPs
└── Llama-3.2-1B-Instruct/              # tokenizer + (large) safetensors
```

## Usage

```bash
# Default prompt
python llama3.2_1b_test.py
python llama3.2_1b_run_from_bin.py

# Custom prompt (uses the model's chat template)
python llama3.2_1b_test.py         --prompt "What is 2+2?"
python llama3.2_1b_run_from_bin.py --prompt "What is 2+2?"

# Pick a DMA device / clock period
python llama3.2_1b_test.py         --dev xdma0 --cycle 5.88
python llama3.2_1b_run_from_bin.py --dev xdma0 --cycle 5.88
```

Both scripts accept `--prompt`, `--dev`, `--cycle`.

## First-run vs cached-run cost

| Step | First `test.py` run | Subsequent `test.py` / any `run_from_bin.py` |
|---|---|---|
| HF tokenizer/safetensors download | ~3 min | skipped |
| Build `weights_*.bin` | ~30 s | skipped |
| Compile unified instruction bin (PBI, 8 prefill + 8 decoder buckets) | ~3 min | skipped — bin loaded from disk |
| Prefill + decode | ~30 s | ~30 s |

## Forcing a rebuild

```bash
# Just the instruction bin (e.g. after editing config buckets)
rm llama3.2_1b_bin/llama3.2_1b_instruction.bin llama3.2_1b_bin/llama3.2_1b_instruction.json

# The weight bin (re-quantizes from the cached HF model)
rm llama3.2_1b_bin/weights_llama3.2_1b_hf.bin

# Full clean (also re-downloads from HF)
rm -rf llama3.2_1b_bin
```

## Expected output (default prompt)

```
--- Starting prefill (actual 44 tokens, bucket 64 = seq_len 63) ---
Prompt tokens (45): (128000, 128006, 9125, 128007, ...)
Prompt text: '<|begin_of_text|>...x+3=5, what is x?...'
Execute program start at 0xD0000000
    Total program execution latency = 6.7e6 us
    GFLOPS: 18.68
Prefill execute done in 6.4 seconds, start decoding...

--- Starting decoder ---
To find the value of x, we need to isolate x on one side of the equation.
...
So, the value of x is 2.
Stop token 128009 reached.

--- Decode benchmark ---
  Tokens decoded : ~71
  Throughput     : ~3.2 tokens/s
  Avg per token  : ~310 ms
```
