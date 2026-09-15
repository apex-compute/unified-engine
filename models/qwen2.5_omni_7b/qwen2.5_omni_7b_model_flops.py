"""Architectural FLOP counts for Qwen2.5-Omni-7B.

These are the FLOPs the MODEL requires at its own dimensions -- what any
implementation of this architecture owes, on any hardware -- as distinct from
the FLOPs this engine ISSUES, which every stage already bills at the padded,
tile-aligned and mask-widened shapes the U55 actually runs.

The two answer different questions.  Issued FLOPs over time is how well the
engine is fed: it is the number to compare against peak.  Model FLOPs over the
same time is the EFFECTIVE rate -- useful work per second -- which is what
compares across implementations, across accelerators, and against another
model.  Their ratio is the fraction of issued work that was useful, so a stage
that looks near peak while running mostly padding shows up here immediately.

Conventions, applied uniformly to keep the counts comparable:

* one multiply-accumulate is 2 FLOPs;
* only matrix products count -- the projections and the two attention matmuls.
  Norms, softmax, activations, RoPE, pooling and residual adds are elementwise
  and are excluded, which is the usual model-FLOPs convention;
* shapes are the LOGICAL ones: the true prompt length rather than the 64-row
  execution multiple, true attention windows rather than a widened mask, and
  the architecture's own vocabulary.

Everything is derived from the model config, so these counts follow the config
rather than drifting from it.
"""

from __future__ import annotations

from collections.abc import Sequence


def gemm(m: int, k: int, n: int) -> int:
    """FLOPs for an m x k by k x n matrix product."""
    return 2 * int(m) * int(k) * int(n)


def _attention(tokens_per_window: Sequence[int], query_dim: int) -> int:
    """Scores + context for one attention layer.

    Each is n x n x head_dim per head, i.e. n^2 x query_dim summed over heads,
    so a layer costs 4 * query_dim * sum(n^2).  Windowed attention pays only
    its own windows; a mask that widens them to the full sequence is an
    implementation cost, not a model one.
    """
    return 4 * int(query_dim) * sum(int(n) * int(n) for n in tokens_per_window)


def vision_flops(cfg: dict, patches: int | None = None,
                 merged_tokens: int | None = None) -> int:
    """One image through the ViT tower: patch embed, 32 blocks, merger."""
    v = cfg["vision"]
    hidden = int(v["hidden_size"])
    inter = int(v["intermediate_size"])
    depth = int(v["depth"])
    merge = int(v["spatial_merge_size"])
    patch = int(v["patch_size"])
    temporal = int(v["temporal_patch_size"])
    vs = int(patches if patches is not None else v["num_patches"])
    merged = int(merged_tokens if merged_tokens is not None
                 else v["num_merged_tokens"])

    # Patch projection: each row is one flattened patch cube.
    total = gemm(vs, 3 * temporal * patch * patch, hidden)

    # Window geometry, in merged units, exactly as the encoder reorders them.
    units = vs // (merge * merge)
    side = int(round(units ** 0.5))
    window_units = int(v["window_size"]) // merge // patch
    windows_per_side = -(-side // window_units)          # ceil
    window_patches = (window_units * merge) ** 2
    windowed = [window_patches] * (windows_per_side * windows_per_side)
    full_layers = set(int(i) for i in v["fullatt_block_indexes"])

    for layer in range(depth):
        total += 4 * gemm(vs, hidden, hidden)            # q, k, v, o
        total += 3 * gemm(vs, hidden, inter)             # SwiGLU gate/up/down
        total += _attention([vs] if layer in full_layers else windowed, hidden)

    # PatchMerger: (hidden * merge^2) -> same -> LM hidden.
    merged_dim = hidden * merge * merge
    total += gemm(merged, merged_dim, merged_dim)
    total += gemm(merged, merged_dim, int(v["out_hidden_size"]))
    return total


def audio_flops(cfg: dict, conv1_rows: int, encoder_states: int,
                chunk_states: Sequence[int], pooled_tokens: int) -> int:
    """One audio item: two convolutions, 32 blocks, the output projection.

    ``chunk_states`` is the post-convolution state count of each 200-mel-frame
    chunk.  Attention is confined to a chunk, so the chunks are the windows.
    """
    a = cfg["audio"]
    hidden = int(a["hidden_size"])
    ffn = int(a["intermediate_size"])
    depth = int(a["depth"])
    states = int(encoder_states)

    total = gemm(int(conv1_rows), 3 * int(a["num_mel_bins"]), hidden)
    total += gemm(states, 3 * hidden, hidden)
    for _layer in range(depth):
        total += 4 * gemm(states, hidden, hidden)        # q, k, v, o
        total += gemm(states, hidden, ffn)               # fc1
        total += gemm(states, ffn, hidden)               # fc2
        total += _attention(chunk_states, hidden)
    total += gemm(int(pooled_tokens), hidden, int(a["out_hidden_size"]))
    return total


def _lm_layer_flops(cfg: dict, tokens: int, kv_lengths: Sequence[int]) -> int:
    """Projections for ``tokens`` rows plus attention over ``kv_lengths``."""
    f = cfg["file_info"]
    hidden = int(f["hidden_size"])
    head_dim = int(f["head_dim"])
    kv_dim = int(f["num_kv_heads"]) * head_dim
    query_dim = int(f["num_kv_heads"]) * int(f["group_size"]) * head_dim
    inter = int(f["mlp_elements"])
    layers = int(f["num_layers"])

    per_layer = gemm(tokens, hidden, query_dim)          # q
    per_layer += 2 * gemm(tokens, hidden, kv_dim)        # k, v
    per_layer += gemm(tokens, query_dim, hidden)         # o
    per_layer += 2 * gemm(tokens, hidden, inter)         # gate, up
    per_layer += gemm(tokens, inter, hidden)             # down
    # Scores and context: every query row attends its own KV history.
    per_layer += 4 * query_dim * sum(int(n) for n in kv_lengths)
    return layers * per_layer


def prefill_flops(cfg: dict, seq_len: int) -> int:
    """Prefill fills the KV cache and stops -- there is no LM head here.

    Causal attention: row i attends i + 1 keys, so the KV lengths sum to
    n(n + 1)/2 rather than n^2.
    """
    n = int(seq_len)
    return _lm_layer_flops(cfg, n, [n * (n + 1) // 2])


def decode_step_flops(cfg: dict, context_len: int) -> int:
    """One greedy step at ``context_len`` KV rows, including the LM head."""
    total = _lm_layer_flops(cfg, 1, [int(context_len)])
    total += gemm(1, int(cfg["file_info"]["hidden_size"]),
                  int(cfg["file_info"]["embedding_vocab"]))
    return total


def decode_flops(cfg: dict, context_lengths: Sequence[int]) -> int:
    """Every step of a decode run, priced at the KV length it actually saw."""
    return sum(decode_step_flops(cfg, n) for n in context_lengths)
