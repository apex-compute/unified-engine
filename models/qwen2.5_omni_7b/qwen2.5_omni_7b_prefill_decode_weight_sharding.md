# Qwen2.5-Omni 7B Weight Sharding

This document describes how the eight-engine runtime uses weights during
prefill and decode. `M` is the number of tokens, `H` is the hidden size, `QH`
is the number of query heads, `KVH` is the number of KV heads, and `N` is a
projection output width.

Every projection weight (Q, K, V, O, gate, up, down) is now **tensor-parallel
in both phases**: one private column shard per engine, staged once at
weight-load time directly into each engine's own window
(`_stage_private_lm_projection`), and reused as-is by prefill and decode --
no second runtime copy, no on-the-fly quantization. V stays BF16 in both
phases (never quantized).

## Prefill

Prefill has `M > 1`. Every projection is column-sharded (tensor-parallel):
engine `e` computes **all `M` rows** of its private `N/8` output-column
shard, using its own private weight block; only the input activation
(`LM_PRE_NORM` / `LM_ATTN_RESULT`) and RMSNorm are row-sharded, since those
are the only row-independent, all-engines-need-every-row steps.

| Operation | Matrix or data shape | Sharding axis | Weight location and use | Same physical weight in prefill and decode? |
|---|---|---|---|---|
| RMSNorm (norm1) | `[M,H]` | Rows: `M_i` | Shared norm parameters | Yes |
| Q projection | `[M,H] x [H,QH*128]` | Output columns: `QH*128/8` per core | Private Q column shards | Yes |
| K projection | `[M,H] x [H,KVH*128]` | Output columns: `KVH*128/8` per core | Private K column shards | Yes |
| V projection | `[M,H] x [H,KVH*128]` | Output columns: `KVH*128/8` per core | Private V column shards (BF16) | Yes |
| RoPE | Q/K activation planes | Token/head planes | No weight | N/A |
| Attention | Q heads attend to KV cache | Query-head sharding; 28 Q heads over 4 KV heads | KV cache and activations | N/A |
| O projection | `[M,QH*128] x [QH*128,H]` | Output columns: `H/8` per core | Private O column shards | Yes |
| Residual and RMSNorm (norm2) | `[M,H]` | Rows: `M_i` | Shared norm parameters | Yes |
| MLP gate | `[M,H] x [H,MLP]` | Output columns: `MLP/8` per core | Private gate N-shards | Yes |
| MLP up | `[M,H] x [H,MLP]` | Output columns: `MLP/8` per core | Private up N-shards | Yes |
| SwiGLU multiply | `[M,MLP/8]` per core | Same local lane | No weight | N/A |
| MLP down | `[M,MLP/8] x [MLP/8,H]` per core | K/reduction lanes: `MLP/8` input columns per core | Private down K-shards | Yes |
| MLP down reduction | Eight partial `[M,H]` results | Cross-core reduction | No weight | N/A |

Column-sharded output still lands in the same true row-major `[M, N]`
buffers (`LM_Q`/`LM_K`/`LM_V`/`LM_ATTN_PROJ`) RoPE/permute/attention already
expect: each engine's `[M, N/8]` block is written at its column offset with
the *full* `N` row stride (`gpr_out_row_stride_reg`), not `N/8` -- so
downstream consumers never need to know the matmul itself was column-split.
(Gate/up/down are the one exception: they use separate per-engine planes,
joined only by `down`'s `reduce_add`, since nothing reads their intermediate
buffers as one true `[M, MLP]` image.)

Token embedding for the prompt's `M` tokens is a one-time host gather done
once before the layer loop starts, not a per-layer op this table covers (see
Decode below, where it repeats every step). Prefill never computes logits or
runs an LM head at all.

## Decode

Decode has `M = 1`. Every projection uses the exact same private column
shards prefill's tensor-parallel path already computed against -- decode
just reads them again, no separate copy or precision conversion.

| Operation | Matrix or data shape | Sharding axis | Weight location and use | Same physical weight in prefill and decode? |
|---|---|---|---|---|
| Q projection | `[1,H] x [H,QH*128]` | Output columns: 8 shards | Private Q column shards | Yes |
| K projection | `[1,H] x [H,KVH*128]` | Output columns: 8 shards (64 columns/core) | Private K column shards | Yes |
| V projection | `[1,H] x [H,KVH*128]` | Output columns: 8 shards (64 columns/core) | Private V column shards (BF16) | Yes |
| Decode attention | One token against KV cache | GQA group/head sharding | KV cache and activations | N/A |
| O projection | `[1,QH*128] x [QH*128,H]` | Output columns: 8 shards (448 columns/core) | Private O column shards | Yes |
| MLP gate | `[1,H] x [H,MLP]` | Output columns: 8 shards | Private gate N-shards | Yes |
| MLP up | `[1,H] x [H,MLP]` | Output columns: 8 shards | Private up N-shards | Yes |
| SwiGLU multiply | `[1,MLP/8]` per core | Same local lane | No weight | N/A |
| MLP down | `[1,MLP/8] x [MLP/8,H]` per core | K/reduction lanes: `MLP/8` input columns per core | Reuses prefill's private down K-shards | Yes |
| MLP output reduction | Eight partial `[1,H]` results | Cross-core sum, then residual add | No weight | N/A |
| Token embedding | One token ID -> `[1,H]` BF16 feature | None across FPGA cores; host gathers one row | Same host BF16 table; one row DMA to `LM_IO_A` per step | Yes, same host table |
| LM head / logits | `[1,H] x [H,Vocab]` | Vocabulary/output columns | Private LM-head column shards | N/A; prefill does not compute logits |

## DRAM and Weight Reuse

Each core owns a 1 GiB DRAM window. The current map reserves:

```text
64 MiB   ISA/program memory
16 MiB   private tensor window
645 MiB  private weight-shard reserve
299 MiB  remaining shared-pool capacity
```

The private weight reserve now covers Q/K/V/O in addition to gate/up/down
and the LM head -- every projection weight lives here exactly once. It no
longer covers IF8 embedding row shards (host-resident) or a second
decode-only copy of anything.

```text
allocate_params_dram() -> shared weights, allocated downward
allocate_tensor_dram() -> KV cache, activations, and scratch, allocated upward
```

The software treats the eight gaps as a distributed pool for placement, but a
single weight or tensor must remain contiguous inside one core window.

Every projection weight is now single-copy, reused by both phases:

| Weight class | Prefill | Decode | Same physical DRAM data? |
|---|---|---|---|
| Q/K/V/O | Private column shards (tensor-parallel) | Same private column shards | Yes |
| MLP gate/up | Private N-shards | Same private N-shards | Yes |
| MLP down | Private K/reduction shards | Same private K/reduction shards | Yes |
| Embedding | BF16 table in host RAM; selected rows DMA to `LM_IO_A` | Same host table; one row DMA per token | No FPGA weight copy in either phase |
| LM head | Not used in prefill | Private vocabulary-column shards | N/A |
| Norms/biases | Shared read-only parameters | Shared or shard-offset read-only parameters | Depends on the parameter and phase |

Vision, audio, and LM shared sections time-share the shared pool. They are
released at phase boundaries before the next phase stages its sections.
