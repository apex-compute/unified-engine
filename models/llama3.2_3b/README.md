# LLaMA 3.2 3B

LLaMA 3.2 3B prefill + decode on the accelerator. Same architecture family as
1B but with 28 layers, hidden=3072, head_dim=128, 8 KV heads, group_size=3,
mlp=8192.

## Files

| File | Purpose |
|---|---|
| `llama3.2_3b_test.py`         | First-run setup: downloads the HF model, generates weight bin + unified PBI instruction bin, then runs. Subsequent runs reuse the cached bins. |
| `llama3.2_3b_run_from_bin.py` | Runtime-only: loads the pre-built bins and runs. No HuggingFace access, no recompile. Fails loudly with a clear error if any required file is missing. |
| `llama3.2_3b_config.json`     | Model dims, bucket lists, weight-bin region offsets. |
| `llama3.2_3b_bin/`            | Generated artifacts (created on first run). |

After the first successful `llama3.2_3b_test.py` run, `llama3.2_3b_bin/` contains:

```
llama3.2_3b_bin/
├── weights_llama3.2_3b_hf.bin          # IF4 weights + bf16 embedding (~2.3 GiB)
├── llama3.2_3b_instruction.bin         # unified PBI instruction bin (~390 MB)
├── llama3.2_3b_instruction.json        # per-bucket offsets + FLOPs
└── Llama-3.2-3B-Instruct/              # tokenizer + (large) safetensors
```

## Usage

```bash
# Default prompt
python llama3.2_3b_test.py
python llama3.2_3b_run_from_bin.py

# Custom prompt (uses the model's chat template)
python llama3.2_3b_test.py         --prompt "What is 2+2?"
python llama3.2_3b_run_from_bin.py --prompt "What is 2+2?"

# Pick a DMA device / clock period
python llama3.2_3b_test.py         --dev xdma0 --cycle 5.88
python llama3.2_3b_run_from_bin.py --dev xdma0 --cycle 5.88
```

Both scripts accept `--prompt`, `--dev`, `--cycle`.

## FPGA DRAM layout

3B uses a custom DRAM layout (set in the engine `__init__`) because the
default 768 MB params region is too small for 3B's ~1.7 GB weights:

| Region | Base | Size | Typical use |
|---|---|---|---|
| Params  | 0x00000000 | 2 GiB    | ~1.7 GB (weights + embedding + rope tables) |
| Tensor  | 0x80000000 | 512 MiB  | ~140 MB (KV cache + activations) |
| Program | 0xA0000000 | 1.5 GiB  | ~400 MB (unified instruction bin) |

Total used: ~2.3 GB out of 4 GB FPGA DRAM.

## First-run vs cached-run cost

| Step | First `test.py` run | Subsequent `test.py` / any `run_from_bin.py` |
|---|---|---|
| HF tokenizer/safetensors download | ~6 min | skipped |
| Build `weights_*.bin` | ~30 s | skipped |
| Compile unified instruction bin (PBI, 8 prefill + 8 decoder buckets) | ~6 min | skipped — bin loaded from disk |
| Prefill + decode | ~75 s | ~75 s |

## Forcing a rebuild

```bash
# Just the instruction bin (e.g. after editing config buckets)
rm llama3.2_3b_bin/llama3.2_3b_instruction.bin llama3.2_3b_bin/llama3.2_3b_instruction.json

# The weight bin (re-quantizes from the cached HF model)
rm llama3.2_3b_bin/weights_llama3.2_3b_hf.bin

# Full clean (also re-downloads from HF)
rm -rf llama3.2_3b_bin
```

## Expected output (default prompt)

```
--- Starting prefill (actual 44 tokens, bucket 64 = seq_len 63) ---
Prompt tokens (45): (128000, 128006, 9125, 128007, ...)
Prompt text: '<|begin_of_text|>...x+3=5, what is x?...'
Execute program start at 0xA0000000
    Total program execution latency = 1.85e7 us
    GFLOPS: 19.41
Prefill execute done in 17.7 seconds, start decoding...

--- Starting decoder ---
To solve for x, we need to isolate x on one side of the equation.
...
So, the value of x is 2.
Stop token 128009 reached.

--- Decode benchmark ---
  Tokens decoded : ~72
  Throughput     : ~1.3 tokens/s
  Avg per token  : ~780 ms
```

Decode per-token time is ~3× slower than 1B (~780 ms vs ~310 ms): each token
does ~3× more weight reads across the larger layer count and wider
projections.
