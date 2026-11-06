# Gemma3 example

This folder contains the Gemma3 accelerator inference example and numeric verification.

> [!WARNING]
> `gemma3_test_IF8.py` is deprecated and IF8 is currently not working. It is
> retained only for historical/debugging reference, excluded from
> `model_auto_test.py`, and must not be treated as a supported inference path.
> Use `gemma3_test.py` (IF4).

## Layout

- **gemma3_test.py** – Prefill + decode loop on accelerator (single or multi engine, `--multi-core N`; both phases shard).
- **../../multi_engine_shard.py** – the shared multi-engine library. The decoder's
  N-sharded matmuls, batch-split attention and master/worker rendezvous all come
  from here (`MultiEngineScheduler`). `model_multicore_layout()` selects disjoint
  model storage and private weight windows for each supported board.
- **gemma3_test_IF8.py** – **Deprecated and currently non-working** IF8 experiment.
- **gemma3_numeric.py** – Numeric verification with torch reference (prefill + decoder).
- **gemma3_config.json** – Model and layout config.
- **decoder_program.json** – Decoder program metadata (written on first decoder compile).
- **gemma3_bin/** – Weights, HF model, and decoder binaries. Contains:
  - `weights_gemma3_hf.bin` or `full_model_weights.bin`
  - `gemma-3-1b-it/` (Hugging Face model, or set via config)
  - `decoder_program.bin`

## Prerequisites

- Run from the **repository root**:
  ```bash
  python models/gemma3/gemma3_test.py
  ```
- Python with `torch`, `transformers`, and DMA device access.

## Usage

From the repository root:

```bash
# Prefill + decode (default prompt)
python models/gemma3/gemma3_test.py

# Custom prompt
python models/gemma3/gemma3_test.py --prompt "Your prompt here"

# DMA device; clock, AXI width, memory size, and core count come from HW_INFO
python models/gemma3/gemma3_test.py --dev xdma1

# Use local full-model weights bin
python models/gemma3/gemma3_test.py --local-weights

```

## Multi-core (`--multi-core N`)

The quantized matmuls of both prefill and decode are N-sharded (split by output
columns) across up to 12 engines, bounded by the core count reported by `HW_INFO`.
Each engine holds its own column block in a private window selected for that
board's memory controllers. Engines synchronize each sharded round through the
four-phase `FLAG_CHECK_SET` / `FLAG_CHECK_CLEAR` handshake. Prefill reads the same
private column blocks as decode, so neither phase adds a second copy of a weight;
see [Prefill](#prefill) below.

```bash
# P2 Kintex-7: two engines on separate DDR controllers
python models/gemma3/gemma3_test.py --dev xdma1 --multi-core 2

# Alveo U50: eight engines using the HBM controller map
python models/gemma3/gemma3_test.py --dev xdma0 --multi-core 8
```

Device numbers depend on the host; the examples above match the tested P2
configuration. Board selection uses the reported hardware signature even when
only two engines of an Alveo board are active.

| Board signature | Shared model range | Private weight windows |
|---|---|---|
| Kintex-7, 2 cores / 4 GiB | `[1, 3)` GiB | 512 MiB at 0 and 3 GiB, on separate DDR controllers |
| U50, 8 cores / 8 GiB | `[6, 8)` GiB | Primary 512 MiB HBM segments in hardware engine/SAXI order |
| U55C, 12 cores / 16 GiB | `[6, 8)` GiB | One 1 GiB controller region per engine; a reserved region moves to the same MC on the other stack when available |
| U55C, 12 cores / 8 GiB | `[6, 8)` GiB | Up to six 1 GiB windows; 7–12 engines use disjoint 512 MiB segments and share some controllers |

The shared model keeps its original internal offsets within a reserved 2 GiB
span: weights at the base, tensors at `+0x30000000`, and instructions at
`+0x50000000`. Allocations and instruction writes are checked against these
bounds before they can overwrite a private shard. Single-engine runs retain the
original addresses.

The raw Kintex memory benchmark uses windows 2 GiB apart. Model inference uses
windows at 0 and 3 GiB so the full shared model fits between them; both choices
place the two private shards on different DDR controllers. The U55C 8 GiB image
cannot provide twelve independent controller regions alongside the model.

Compiled multi-engine filenames include a hash of the model base and ordered
private windows, so a binary from another placement cannot be reused accidentally.
`--bin-reuse` remains available for single-engine runs. Multi-engine runs always
recompile the primary image and its persistent worker programs together.

### Measured validation

Both comparisons used the same 28-token prefill and produced exactly the same
70 decoded token IDs, including the stop token, and decoded text as their
single-engine baseline. The prompt was `Solve 2x + 3 = 7. Reply with only the value
of x.` First-token rates below use hardware execution time.

| Board | Engines | First-token rate, single → multi | First-token speedup | Mean hardware decode time, single → multi |
|---|---:|---:|---:|---:|
| Kintex-7, 198.324 MHz | 1 → 2 | 10.60 → 19.64 tok/s | 1.85× | 95.77 → 52.16 ms/token |
| U50, 333.332 MHz | 1 → 8 | 16.83 → 76.80 tok/s | 4.56× | 60.28 → 13.11 ms/token |

See the [Kintex-7 results](../../tests/kintex7_gemma3_controller_results.json) and
[U50 results](../../tests/alveo_u50_gemma3_controller_results.json) for the complete
hardware signatures, prompt, token IDs, placements, and timing measurements.
These are end-to-end decode measurements, not a claim that every model operation
sustains the raw memory benchmark's bandwidth. U55C placement is covered by
software tests; these results do not include a U55C hardware run.

### What is sharded

Optional shards are controlled by the flags in `gemma3_test.py`:

| op | K | N | 8-way split | flag |
|---|---:|---:|---|---|
| Q proj | 1152 | 1024 | 128 x 8 (even) | `SHARD_QKV` |
| K / V proj | 1152 | 256 | **not splittable past 4 engines** (4 blocks of 64) | `SHARD_QKV` |
| attn O proj | 1024 | 1152 | 128 x 6 + 192 x 2 | `SHARD_ATTN_OPROJ` |
| MLP gate | 1152 | 6912 | 832 x 4 + 896 x 4 | always on |
| MLP up | 1152 | 6912 | 832 x 4 + 896 x 4 | always on |
| MLP down | 6912 | 1152 | 128 x 6 + 192 x 2 | `SHARD_MLP_DOWN` |
| LM head | 1152 | 262144 | 32768 x 8 (even) | always on |

A column shard must be a whole multiple of `UE_VECTOR_SIZE` (64), so remainders are
handed to the trailing engines and engine 0 -- which also runs everything unsharded --
takes the smallest block. Ops that share an input and write disjoint outputs ride in one
round: Q/K/V together, and gate+up together.

### Prefill

Prefill stays one seq_len-agnostic program: a single layer body, hardware-looped 26
times, whose row count is a runtime register. Splitting rows across engines would bake
the prompt length into each engine's slice, and a 16-token prompt would leave each
engine two rows padded to a 64-row tile. Prefill therefore splits COLUMNS, like decode:
every engine computes all rows of its own column block and writes it in place into the
shared full-width output (a strided write-back, `gpr_out_row_stride_reg` on
`quantized_matmat_core`). Each engine runs its own looped worker program
(`MultiEngineScheduler.emit_rows_worker_programs`), meeting engine 0 four times per layer:

| Round | Ops | Weights |
|---|---|---|
| QKV | Q, K, V projections (V straight into the KV cache) | Q: decode's private shards. K/V: their eight 64-column blocks dealt out from the shared image, so at 8 engines engines 0-3 take a K block and 4-7 a V block (192 columns per engine) |
| O | attention output projection | decode's private shards |
| gate/up | gate (GELU) + up | decode's private shards |
| down | MLP down | decode's private shards |

The RMS norms, RoPE, attention and the gate*up multiply between the rounds stay on
engine 0. `--legacy` and `--two-pass-prefill` keep prefill on engine 0.

Measured on the U50 at 333.332 MHz with `--multi-core 8` (8-engine decode in both
columns; the "before" column is prefill on engine 0). Every prompt decoded exactly the
same text as the single-engine-prefill run. One engine's peak is 42.7 GFLOPS, eight
engines' is 341.3.

| Prompt | Prefill tokens | Prefill HW time, before → after | Prefill GFLOPS, before → after | Speedup |
|---|---:|---:|---:|---:|
| `What is the capital of France?` | 16 | 574.8 → 108.4 ms | 39.5 → 209.1 | 5.30× |
| `Solve 2x + 3 = 7. Reply with only the value of x.` | 28 | 995.0 → 180.2 ms | 39.9 → 220.2 | 5.52× |
| sky-is-blue paragraph request | 72 | 2555.2 → 462.7 ms | 40.0 → 220.5 | 5.52× |
| ten "Fact i" sentences + question | 169 | 6005.5 → 1095.3 ms | 39.9 → 218.7 | 5.48× |

Profile of the 16-token prompt (`--profile`, summed over the 26 layers): the four sharded
steps reach 54-84% of the 8-engine peak (gate/up 286 GFLOPS, 84%). Attention, still on
engine 0, is now the largest single-engine step at 18.0 of 108.5 ms.

| Step | ms | GFLOPS | Engines |
|---|---:|---:|---:|
| pre_norm | 0.50 | 3.9 | 1 |
| qkv_proj_vcache | 5.70 | 258.5 | 8 |
| qk_norm_rope | 2.86 | 1.5 | 1 |
| attention | 17.95 | 6.1 | 1 |
| o_proj_post_attn_norm_residual | 5.37 | 183.4 | 8 (norm/residual on 1) |
| pre_ffn_norm | 0.44 | 4.3 | 1 |
| mlp_gateup_gelu_mul | 46.26 | 286.5 | 8 (multiply on 1) |
| mlp_down_post_ffn_norm_residual | 29.41 | 225.4 | 8 (norm/residual on 1) |
| **Total** | **108.48** | **207.0** | |
