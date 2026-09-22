"""Architectural FLOP counts for Gemma-4 E2B.

These are the FLOPs the MODEL requires at its own dimensions -- what any
implementation of this architecture owes, on any hardware -- as distinct from
the FLOPs this engine ISSUES, which every stage already bills at the padded,
tile-aligned and mask-widened shapes the accelerator actually runs. See
qwen2.5_omni_7b_model_flops.py for the fuller rationale; the convention here
is identical:

* one multiply-accumulate is 2 FLOPs;
* only matrix products count -- projections and the two attention matmuls.
  Norms, softmax, activations (gelu/SiLU), RoPE, pooling, elementwise
  gate/scale and residual adds are excluded, the usual model-FLOPs convention;
* shapes are the LOGICAL ones: the true prompt/context length, not the
  64-row execution multiple or a widened mask.

GEMMA-4 E2B IS NOT UNIFORM ACROSS LAYERS, which is what makes this module more
than a straight port of qwen's. Two axes vary per layer, both read from
``cfg["model"]``:

  attention type   every 5th layer (``full_attention_layers``) is GLOBAL --
                    head_dim=512, attends the WHOLE causal history, like a
                    standard decoder layer. The other 28 are LOCAL/SLIDING --
                    head_dim=256, attention capped to the most recent
                    ``sliding_window`` (512) tokens, regardless of how long
                    the context has grown. Both use group_size=8 query heads
                    sharing ONE KV head (MQA), so query width is
                    head_dim * group_size and KV width is head_dim.

  MLP width         layers below ``double_wide_mlp_first_layer`` (15) use
                    ``mlp_elements`` (6144); layers at or above it use
                    ``mlp_elements_wide`` (12288).

PER-LAYER INPUT INJECTION (PLE) is real matmul work this architecture has that
qwen's does not: one M x hidden x (num_layers * per_layer_dim) projection
ONCE per forward pass, plus a per-layer gate (hidden -> per_layer_dim) and
proj (per_layer_dim -> hidden) matmul pair EVERY layer. The elementwise
gather/multiply/scale/norm/residual steps around it are excluded as usual.

Vision is a standard, non-windowed 16-layer SigLIP-style tower (no per-layer
variation, no sliding attention) -- the simple case, structured the same way
as qwen's vision_flops.
"""

from __future__ import annotations

from collections.abc import Sequence

# Vision-encoder constants, matching Gemma4VisionMixin's class attributes
# (gemma4_e2b_vision.py). Not in the JSON config -- these are fixed by the
# vision tower's architecture, not a per-run choice.
VIS_H = 768
VIS_MLP = 3072
VIS_LAYERS = 16


def gemm(m: int, k: int, n: int) -> int:
    """FLOPs for an m x k by k x n matrix product."""
    return 2 * int(m) * int(k) * int(n)


def _causal_windowed_kv_sum(n: int, window: int | None) -> int:
    """sum_{i=1}^{n} min(i, window) -- total keys attended across n causal rows.

    ``window=None`` is unbounded causal attention: row i attends i keys, sum
    is the familiar n(n+1)/2. A window caps every row at ``window`` keys once
    i exceeds it, which is exactly a sliding-attention layer's prefill cost
    (and generalizes correctly even though this implementation's prefill is
    currently capped at seq_len <= sliding_window, where the two formulas
    agree).
    """
    n = int(n)
    if window is None or n <= window:
        return n * (n + 1) // 2
    w = int(window)
    return w * (w + 1) // 2 + w * (n - w)


def vision_flops(num_patches: int, soft_tokens: int, text_hidden: int = 1536) -> int:
    """One image through the vision tower: patch proj, 16 layers, output proj.

    Full (non-windowed) self-attention over every patch, every layer -- no
    per-layer variation, unlike the LM. ``soft_tokens`` is the count after
    average pooling, which is not a matmul and is excluded. ``text_hidden``
    is VIS_TEXT_H, the LM hidden the final projection lands in -- a runtime
    value read from the compiled program's metadata (gemma4_e2b_vision.py),
    not a fixed constant like VIS_H/VIS_MLP/VIS_LAYERS, so it is a parameter
    here rather than baked in; 1536 matches that code's own default.
    """
    S = int(num_patches)
    H, mlp, layers = VIS_H, VIS_MLP, VIS_LAYERS

    total = gemm(S, H, H)                          # patch embed refinement
    for _layer in range(layers):
        total += 4 * gemm(S, H, H)                  # q, k, v, o (standard MHA)
        total += 2 * gemm(S, H, mlp)                 # gate, up
        total += gemm(S, mlp, H)                      # down
        total += 4 * H * S * S                        # full self-attention
    total += gemm(int(soft_tokens), H, int(text_hidden))  # pooled -> LM hidden
    return total


def _layer_dims(cfg: dict, layer_idx: int) -> tuple[int, int, int]:
    """(head_dim, query_width, kv_width) for one layer -- mirrors
    _get_layer_attention_dims in gemma4_e2b_lm.py exactly."""
    f = cfg["file_info"]
    m = cfg["model"]
    group_size = int(f["group_size"])
    if layer_idx in set(int(i) for i in m["full_attention_layers"]):
        head_dim = int(f["head_dim"])
    else:
        head_dim = int(f["head_dim_sliding"])
    return head_dim, head_dim * group_size, head_dim


def _mlp_width(cfg: dict, layer_idx: int) -> int:
    f, m = cfg["file_info"], cfg["model"]
    if layer_idx >= int(m["double_wide_mlp_first_layer"]):
        return int(f["mlp_elements_wide"])
    return int(f["mlp_elements"])


def _is_global(cfg: dict, layer_idx: int) -> bool:
    return layer_idx in set(int(i) for i in cfg["model"]["full_attention_layers"])


def _per_layer_input_flops(cfg: dict, tokens: int) -> int:
    """The ONE model-proj matmul plus each layer's gate/proj pair.

    Symmetric between prefill (tokens=seq_len, once) and decode (tokens=1,
    once per step) -- both call the same shapes in gemma4_e2b_lm.py.
    """
    f = cfg["file_info"]
    hidden = int(f["hidden_size"])
    dim = int(f["per_layer_input_dim"])
    layers = int(f["num_layers"])
    t = int(tokens)

    total = gemm(t, hidden, layers * dim)             # model_proj, once
    per_layer_pair = gemm(t, hidden, dim) + gemm(t, dim, hidden)  # gate, proj
    total += layers * per_layer_pair
    return total


def _attn_and_proj_flops(cfg: dict, tokens: int, kv_sum_by_layer) -> int:
    """Q/K/V/O projections + MLP + attention, summed over every layer.

    ``kv_sum_by_layer(layer_idx, is_global)`` returns the total keys attended
    across ``tokens`` query rows for that layer -- the caller supplies it so
    this one function serves both prefill (a causal sum) and decode (a single
    row's KV length).
    """
    f = cfg["file_info"]
    hidden = int(f["hidden_size"])
    layers = int(f["num_layers"])
    t = int(tokens)

    total = 0
    for layer_idx in range(layers):
        head_dim, q_width, kv_width = _layer_dims(cfg, layer_idx)
        mlp = _mlp_width(cfg, layer_idx)

        total += gemm(t, hidden, q_width)              # q
        total += 2 * gemm(t, hidden, kv_width)          # k, v (one KV head)
        total += gemm(t, q_width, hidden)               # o
        total += 2 * gemm(t, hidden, mlp)                # gate, up
        total += gemm(t, mlp, hidden)                     # down

        kv_sum = kv_sum_by_layer(layer_idx, _is_global(cfg, layer_idx))
        total += 4 * q_width * kv_sum                    # scores + context
    return total


def prefill_flops(cfg: dict, seq_len: int) -> int:
    """Prefill fills the KV cache and stops -- no LM head here.

    Causal attention: row i attends i + 1 keys. A global layer sees the whole
    causal history; a sliding layer's row is additionally capped at
    ``sliding_window`` keys, via _causal_windowed_kv_sum.
    """
    n = int(seq_len)
    window = int(cfg["model"]["sliding_window"])

    def kv_sum(_layer_idx: int, is_global: bool) -> int:
        return _causal_windowed_kv_sum(n, None if is_global else window)

    total = _attn_and_proj_flops(cfg, n, kv_sum)
    total += _per_layer_input_flops(cfg, n)
    return total


def decode_step_flops(cfg: dict, context_len: int) -> int:
    """One greedy step at ``context_len`` KV rows, including the LM head.

    A global layer's single query row attends every one of the context_len
    keys; a sliding layer's row is capped at min(context_len, sliding_window)
    -- the same rule gemma4_e2b_lm.py's decode bias construction applies
    (``if self.seq_len <= self.sliding_window: ... else: window_start = ...``).
    """
    c = int(context_len)
    window = int(cfg["model"]["sliding_window"])

    def kv_sum(_layer_idx: int, is_global: bool) -> int:
        return c if is_global else min(c, window)

    total = _attn_and_proj_flops(cfg, 1, kv_sum)
    total += _per_layer_input_flops(cfg, 1)
    total += gemm(1, int(cfg["file_info"]["hidden_size"]),
                  int(cfg["file_info"]["embedding_vocab"]))
    return total


def decode_flops(cfg: dict, context_lengths: Sequence[int]) -> int:
    """Every step of a decode run, priced at the KV length it actually saw."""
    return sum(decode_step_flops(cfg, n) for n in context_lengths)
