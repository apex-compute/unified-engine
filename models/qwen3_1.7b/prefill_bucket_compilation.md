# Prefill Bucket Compilation

## Goal

Pre-compile prefill instruction streams for fixed KV bucket sizes so that `run_from_bin.py` never compiles at runtime — it loads the right bucket's instructions and runs.

## How it works

### Compilation (`qwen3_1.7b_test_clone.py --kv-variants 512`)

1. Sets up the UE (weights + tensor init).
2. For each KV bucket `[64, 128, 192, 256, 320, 384, 448, 512]`:
   - `compile_prefill(seq_len=bucket)` captures instructions into the capture buffer.
   - Reads `self.capture_buffer` to get raw instruction bytes.
   - Calls `reset_program_dram_addr()` so each variant starts at the same DRAM base address.
   - Accumulates the bytes (or saves to individual files).
3. Writes the combined bin + meta.

### Runtime (`qwen3_1.7b_run_from_bin.py`)

1. Prompt length → `actual_prefill_len = len(prompt) - 1`.
2. Pick smallest bucket `>= actual_prefill_len`.
3. Load the bucket's instructions from the bin (via offset into the combined file, or from a per-bucket file).
4. DMA write to program DRAM, execute.

## DRAM layout (unchanged)

Everything stays at the current sizes (`tensor_init` at `MAX_CONTEXT_SIZE=2048`). The compiled programs reference these same large addresses — they only access the first `bucket_len` positions, which fits within the max-sized buffers.

```
Params:    0x00000000 – 0x57FFFFFF  (weights)
Tensors:   0x58000000 – 0x97FFFFFF  (activations, KV cache)
Programs:  0x98000000 – 0xFFFFFFFF  (instructions)
```

## Instruction size scaling

The prefill instruction count grows linearly with seq_len because of per-token loops:

| Bucket | Est. instruction bytes |
|--------|----------------------|
| 64     | 40 MB                |
| 128    | 120 MB               |
| 192    | 240 MB               |
| 256    | 400 MB               |
| 320    | 600 MB               |
| 384    | 830 MB               |
| 448    | 1.1 GB               |
| 512    | 1.4 GB               |

**Combined total: ~4.7 GB**

At runtime only one bucket is loaded (~40 MB–1.4 GB depending on prompt size), so the DRAM budget is fine.

## Options for storage

### Option A: Individual bin files

```
prefill_S64.bin   (40 MB)
prefill_S128.bin  (120 MB)
...
prefill_program.json  -> maps bucket sizes to filenames
```

**Pros**: Only download/extract what you need. Simple to load (`open + dma_write`).
**Cons**: Many files.

### Option B: One combined bin + offset map

```
prefill_program.bin     (4.7 GB)
prefill_program.json    -> { "buckets": [64, 128, ...], "offsets": [0, 41943040, ...] }
```

**Pros**: Single file.
**Cons**: Must allocate/read ~4.7 GB even if only one bucket is needed. DMA write at an offset into program DRAM is straightforward.

## What changed in `compile_prefill`

Added `save_path` parameter. When set, the capture buffer bytes are written to a bin file before `clear_capture_buffer()` is called. The existing DRAM write + allocate still happens (needed for `get_program_dram_addr()` etc.), but the program pointer is reset between variants in the kv-variants loop.
