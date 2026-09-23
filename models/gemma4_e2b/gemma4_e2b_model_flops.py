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


def vision_flops_by_phase(num_patches: int, soft_tokens: int,
                          text_hidden: int = 1536) -> dict[str, int]:
    """Same total as vision_flops, broken out by the compiled program's own
    checkpoint names (gemma4_e2b_vision.py's ``_checkpoint(...)`` calls),
    read from source rather than reverse-engineered from timings:

      patch_embed  the one-time H->H refinement before the layer loop
      proj         q, k, v ONLY -- o is bundled into post_attn, matching
                   where the real _checkpoint("L{i}_proj") actually fires
      rope         0 by convention (see module docstring) -- the real
                   kernel implements it as a swap-matrix matmul, which is a
                   hardware trick, not architecturally required work
      permute      0 -- pure data-layout reshuffle (bf16_permute_dram_core),
                   no matmul
      attention    scores + context, every layer
      post_attn    o_proj + the full gate/up/down MLP, bundled -- matches
                   _checkpoint("L{i}_post_attn") firing after both
      pooler_tail  the final H->text_hidden projection ONLY. The average
                   pool itself is ALSO a real matmul on this hardware (a
                   fixed weight-1 pooling matrix, gemma4_e2b_vision.py's
                   "wide-acc matmul pool" comment) but pooling is
                   conceptually O(N), not a model-required matrix product,
                   so it stays excluded -- the same choice qwen's own
                   merger makes for its pooling step. This is exactly the
                   kind of gap a per-phase breakdown exists to surface, not
                   paper over by inflating the model side to match.
    """
    S = int(num_patches)
    H, mlp, layers = VIS_H, VIS_MLP, VIS_LAYERS

    phases = {
        "patch_embed": gemm(S, H, H),
        "proj": 0, "rope": 0, "permute": 0, "attention": 0, "post_attn": 0,
    }
    for _layer in range(layers):
        phases["proj"] += 3 * gemm(S, H, H)             # q, k, v
        phases["attention"] += 4 * H * S * S
        phases["post_attn"] += (
            gemm(S, H, H)                                # o
            + 2 * gemm(S, H, mlp) + gemm(S, mlp, H)       # gate, up, down
        )
    phases["pooler_tail"] = gemm(int(soft_tokens), H, int(text_hidden))
    return phases


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
    return sum(vision_flops_by_phase(num_patches, soft_tokens, text_hidden).values())


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


def _per_layer_input_phases(cfg: dict, tokens: int) -> tuple[int, int]:
    """(per_layer_prepare, inject) -- the ONE model-proj matmul, and the sum
    of every layer's gate/proj pair, matching gemma4_e2b_lm.py's own
    checkpoint names exactly (``_checkpoint("per_layer_prepare")`` fires
    right after the model-proj matmul and before the layer loop;
    ``_checkpoint(f"L{i}_inject")`` fires after that layer's gate/proj pair).

    Symmetric between prefill (tokens=seq_len, once) and decode (tokens=1,
    once per step) -- both call the same shapes in gemma4_e2b_lm.py.
    """
    f = cfg["file_info"]
    hidden = int(f["hidden_size"])
    dim = int(f["per_layer_input_dim"])
    layers = int(f["num_layers"])
    t = int(tokens)

    per_layer_prepare = gemm(t, hidden, layers * dim)
    per_layer_pair = gemm(t, hidden, dim) + gemm(t, dim, hidden)  # gate, proj
    inject = layers * per_layer_pair
    return per_layer_prepare, inject


def _layer_breakdown(cfg: dict, tokens: int, kv_sum_by_layer) -> dict[str, int]:
    """qkv / o / mlp / attention, each summed over every layer separately.

    Kept as four buckets rather than one total because prefill and decode
    checkpoint these differently: prefill's real _checkpoint(f"L{i}_mlp")
    fires after BOTH o_proj and the gate/up/down MLP (see
    _emit_prefill_post_attention_shard in gemma4_e2b_lm.py, one sharded
    region with no checkpoint between them), so prefill_flops_by_phase
    merges o into mlp; decode's real checkpoints put
    _checkpoint(f"L{i}_o_proj") and _checkpoint(f"L{i}_mlp") on either side
    of the gate/up/down projections, so decode_step_flops_by_phase keeps
    them apart. ``kv_sum_by_layer(layer_idx, is_global)`` returns the total
    keys attended across ``tokens`` query rows for that layer -- the caller
    supplies it so this one function serves both prefill (a causal sum) and
    decode (a single row's KV length).
    """
    f = cfg["file_info"]
    hidden = int(f["hidden_size"])
    layers = int(f["num_layers"])
    t = int(tokens)

    out = {"qkv": 0, "o": 0, "mlp": 0, "attention": 0}
    for layer_idx in range(layers):
        head_dim, q_width, kv_width = _layer_dims(cfg, layer_idx)
        mlp = _mlp_width(cfg, layer_idx)

        out["qkv"] += gemm(t, hidden, q_width) + 2 * gemm(t, hidden, kv_width)
        out["o"] += gemm(t, q_width, hidden)
        out["mlp"] += 2 * gemm(t, hidden, mlp) + gemm(t, mlp, hidden)

        kv_sum = kv_sum_by_layer(layer_idx, _is_global(cfg, layer_idx))
        out["attention"] += 4 * q_width * kv_sum
    return out


def prefill_flops_by_phase(cfg: dict, seq_len: int) -> dict[str, int]:
    """Same total as prefill_flops, broken out by compile_prefill's own
    checkpoint names: per_layer_prepare, qkv_vproj, rope(=0), q_permute(=0),
    attention, mlp (o_proj bundled in, see _layer_breakdown), inject.

    Causal attention: row i attends i + 1 keys. A global layer sees the
    whole causal history; a sliding layer's row is additionally capped at
    ``sliding_window`` keys, via _causal_windowed_kv_sum.
    """
    n = int(seq_len)
    window = int(cfg["model"]["sliding_window"])

    def kv_sum(_layer_idx: int, is_global: bool) -> int:
        return _causal_windowed_kv_sum(n, None if is_global else window)

    layer = _layer_breakdown(cfg, n, kv_sum)
    prepare, inject = _per_layer_input_phases(cfg, n)
    return {
        "per_layer_prepare": prepare,
        "qkv_vproj": layer["qkv"],
        "rope": 0,
        "q_permute": 0,
        "attention": layer["attention"],
        "mlp": layer["o"] + layer["mlp"],
        "inject": inject,
    }


def prefill_flops(cfg: dict, seq_len: int) -> int:
    """Prefill fills the KV cache and stops -- no LM head here."""
    return sum(prefill_flops_by_phase(cfg, seq_len).values())


def decode_step_flops_by_phase(cfg: dict, context_len: int) -> dict[str, int]:
    """Same total as decode_step_flops, broken out by compile_decoder's own
    checkpoint names: per_layer_prepare, qkv_vproj, rope(=0), attention,
    o_proj, mlp, inject, lm_head -- o_proj and mlp are SEPARATE phases here,
    unlike prefill (see _layer_breakdown).

    A global layer's single query row attends every one of the context_len
    keys; a sliding layer's row is capped at min(context_len, sliding_window)
    -- the same rule gemma4_e2b_lm.py's decode bias construction applies
    (``if self.seq_len <= self.sliding_window: ... else: window_start = ...``).
    """
    c = int(context_len)
    window = int(cfg["model"]["sliding_window"])

    def kv_sum(_layer_idx: int, is_global: bool) -> int:
        return c if is_global else min(c, window)

    layer = _layer_breakdown(cfg, 1, kv_sum)
    prepare, inject = _per_layer_input_phases(cfg, 1)
    return {
        "per_layer_prepare": prepare,
        "qkv_vproj": layer["qkv"],
        "rope": 0,
        "attention": layer["attention"],
        "o_proj": layer["o"],
        "mlp": layer["mlp"],
        "inject": inject,
        "lm_head": gemm(1, int(cfg["file_info"]["hidden_size"]),
                        int(cfg["file_info"]["embedding_vocab"])),
    }


def decode_step_flops(cfg: dict, context_len: int) -> int:
    """One greedy step at ``context_len`` KV rows, including the LM head."""
    return sum(decode_step_flops_by_phase(cfg, context_len).values())


def decode_flops(cfg: dict, context_lengths: Sequence[int]) -> int:
    """Every step of a decode run, priced at the KV length it actually saw."""
    return sum(decode_step_flops(cfg, n) for n in context_lengths)
