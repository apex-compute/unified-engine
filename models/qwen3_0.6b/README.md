# Qwen3 0.6B

Qwen3-0.6B text generation on the accelerator. This implementation uses the
official `Qwen/Qwen3-0.6B` checkpoint with adaptive IF4 projection weights and
a packaged BF16 final-layer fallback. IF4 selects INT4 or FP4 per 64-value block
by minimum reconstruction error without changing the packed size.

## Architecture

- 28 transformer layers
- 1024 hidden size and 3072 SwiGLU intermediate size
- 16 query heads, 8 key/value heads, and 128 dimensions per head
- 151,936-token vocabulary with tied input/output embeddings
- 1,000,000 RoPE theta
- Hardware context cap of 4096 tokens and prefill cap of 512 tokens

The query projection is 2048 wide (`16 * 128`) even though the residual hidden
size is 1024. The output projection maps those 2048 attention features back to
the 1024-wide residual stream.

## Files

| File | Purpose |
|---|---|
| `qwen3_0.6b_test.py` | Downloads/caches the checkpoint, builds `params.bin`, compiles `programs.bin`, and runs prefill plus decode. |
| `qwen3_0.6b_run_from_bin.py` | Execute-only offline runner for previously built artifacts. |
| `qwen3_0.6b_config.json` | Model dimensions, artifact paths, and packed-weight layout. |
| `qwen3_0.6b_bin/` | Generated weights, programs, metadata, and tokenizer cache. |

`params.bin` includes the BF16 embedding table, IF4 model weights, and the BF16
final-layer projections. The run-from-bin path therefore needs only the files
listed below; it does not load the full Hugging Face checkpoint.

At startup, the exact 296.75 MiB BF16 embedding table is copied once into the
top of accelerator DRAM. Decode then selects a prebuilt 64-byte dispatch entry
for the latest token; that entry supplies the embedding-row address to the
cached decoder, which copies the row internally into its recurrent input.

## Usage

Run these commands from the repository root:

```bash
# First run: download the checkpoint, build artifacts, and execute.
python3 models/qwen3_0.6b/qwen3_0.6b_test.py \
  --prompt "If x + 3 = 5, what is x?"

# Later runs: execute from local artifacts without network access or compilation.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 models/qwen3_0.6b/qwen3_0.6b_run_from_bin.py \
  --prompt "If x + 3 = 5, what is x?"
```

The offline runner requires:

```text
qwen3_0.6b_bin/params.bin
qwen3_0.6b_bin/params.json
qwen3_0.6b_bin/programs.bin
qwen3_0.6b_bin/programs.json
qwen3_0.6b_bin/Qwen3-0.6B/tokenizer files
```

Delete `programs.bin` and `programs.json` to force instruction recompilation.
Delete the full `qwen3_0.6b_bin` directory to rebuild every artifact.
Program metadata fingerprints the model compiler, shared accelerator kernels,
configuration, execution precision, and weight codec so stale artifacts fail
closed instead of being reused silently.

## Common flags

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | Configured example | Wrapped with the Qwen chat template. |
| `--dev` | `xdma0` | DMA device name. |
| `--device` | `kintex7` | FPGA board profile; use `efinix` for that DMA path. |
| `--local-weights` | Off | Select `full_model_weights.bin` instead of generated `params.bin`. The file must use this config's layout. |
| `--pure-greedy` | Off | Disable the on-FPGA repetition-penalty bias. |
| `--max-new-tokens` | `0` | Stop after N decoded tokens; `0` uses the model stop/context limit. |
| `--bf16-last-layer` | Off | Use the packaged BF16 layer 27 projections. The default fused IF4 path is faster. |
| `--greedy-until` | `512` | Decoded tokens before enabling the repetition penalty. |
| `--pen-alpha` | `1.0` | Per-occurrence logit penalty. |
| `--pen-cap` | `20.0` | Maximum absolute penalty per token. |
| `--rep-window` | `256` | Token-history window used by the penalty. |
| `--cycle` | `5.62` | Offline runner only: clock period in nanoseconds. |

The instruction image is precision-mode specific. To use
`--bf16-last-layer` with the offline runner, first build that image by passing
the same flag to `qwen3_0.6b_test.py`; rerun the test without the flag to restore
the default IF4 image.

## Performance

The Unified Engine board measurements supplied for this comparison are shown
alongside the Qwen result reported in the
[Architect Labs Redwood paper](https://architectlabs.com/architect-labs-redwood.pdf):

| System | FPGA | Clock | Memory bandwidth | Qwen3-0.6B throughput |
|---|---|---:|---:|---:|
| Unified Engine | Kintex-7 | 194 MHz | 6 GB/s | 12.3 tokens/s |
| Unified Engine | Kintex UltraScale+ KU5P | 333 MHz | 9 GB/s | 19.5 tokens/s |
| Unified Engine | Virtex UltraScale+ (Alveo U50) | 366 MHz | 11 GB/s | 21 tokens/s |
| Architect Labs Redwood Nano | AMD Versal VPK180, 2 x 2 tiles | 250 MHz | 16 GB/s | 12.1 tokens/s average; 13 tokens/s peak |

The Kintex-7 result is 1.02x Redwood's reported average throughput while using
37.5% of its listed peak memory bandwidth and running at 77.6% of its clock.
The Kintex UltraScale+ KU5P result is 1.61x Redwood's average and 1.50x its peak;
the Alveo U50 result is 1.74x Redwood's average and 1.62x its peak. These are
indicative comparisons rather than a controlled head-to-head benchmark:
Redwood averages 128 generated tokens and includes sending the prompt to the
FPGA and returning every output token, while the accelerators, runtime paths,
clocks, memory systems, and model packaging differ. The paper's 49 tokens/s
result is an ASIC projection at 1 GHz and about 64 GB/s, not a measured FPGA
result.

The Unified Engine implementation covers more than the throughput comparison:
it provides complete prompt prefill and autoregressive decode, a 4096-token KV
cache, on-FPGA repetition-bias and argmax token selection, an adaptive-IF4
default with a packaged BF16 layer-27 fallback, and fingerprinted artifacts for
offline replay. The complete-model runs validate the deployable path, not only
an isolated operator benchmark.

The decoder uses a one-pass streaming IF4 LM head, shared RoPE-table loads,
single-pass Q scaling, direct per-head attention outputs, recurrent layer
buffers, and one cached preamble per 64-token attention bucket. A 9.27 MiB
token-dispatch table removes the old per-token 2 KiB embedding H2C transfer:

```text
host selects dispatch_base + token_id * 64
    -> ADD_SET TMP_REG = BF16 row word address
    -> cached attention-bucket preamble
    -> cached decoder
    -> device DRAM row copy into LAYER0_INPUT_DRAM
```

Attention-mask updates, and repetition-bias updates after their configured
gate, still cross the host/device boundary.

## Memory layout

- Params DRAM starts at `0x80000000` and uses about 319 MiB in the default IF4
  mode (up to about 349 MiB with the BF16 layer-27 fallback).
- Tensor DRAM starts at `0xA0000000` and uses about 718 MiB at the 4096-token cap,
  including 448 MiB of K/V cache.
- Program DRAM starts at `0xE0000000`; the current 6.12 MiB program image,
  128-byte runtime preamble, and 9.27 MiB token-dispatch table end at
  `0xE0F65D80`.
- The exact BF16 embedding table occupies `0xED740000–0x100000000`
  (296.75 MiB), leaving about 199.85 MiB of guarded space above the current
  program/dispatch allocation. A host copy remains only for prompt prefill;
  token-to-token decode reads embedding rows wholly within device DRAM.

The official checkpoint advertises a longer context, but its K/V cache alone
would exceed the accelerator's 4 GiB address space at that limit. The local
4096-token cap matches the existing Qwen3 accelerator example.
