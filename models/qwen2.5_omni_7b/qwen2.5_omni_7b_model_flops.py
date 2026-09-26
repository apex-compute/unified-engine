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

PER-PHASE BREAKDOWNS (``*_by_phase``) exist alongside the scalar totals so
--profile's per-phase tables can show effective GFLOPS next to issued GFLOPS
for each real phase, not just a whole-stage aggregate. The phase keys are not
invented -- they are the exact checkpoint names the compiled programs use
(``_ckpt(f"L{li}:qkv_proj")`` etc. in qwen2.5_omni_7b_vision.py and
qwen2.5_omni_7b_lm.py, read from source), so a dict here lines up directly with
a row _aggregate_vis_profile prints. Two attribution details worth knowing:

* Vision's patch embedding runs ONCE as a separate FPGA program before the
  checkpointed encoder. Its hardware-counter time and issued FLOPs are
  reported as a separate ``patch_embed`` profile row, so the model FLOPs
  must also stay separate from layer 0's ``qkv_proj`` bucket.
* qkv_proj/o_proj/mlp_gate_up/mlp_proj are real matmuls; rope(+cache),
  permute/unpermute/attn_permute and the norm phases are elementwise and
  priced at 0 by convention -- a real, honest 0% useful for that phase, not
  a missing value.
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


def vision_flops_by_phase(cfg: dict, patches: int | None = None,
                          merged_tokens: int | None = None) -> dict[str, int]:
    """Same total as vision_flops, broken out by the compiled program's own
    checkpoint names (qwen2.5_omni_7b_vision.py's ``_ckpt(...)`` calls), plus
    the separately executed patch_embed program: qkv_proj, permute_qkv
    (=0), rope (=0), attention, unpermute+trim (=0), o_proj+mlp (bundled --
    matches the real checkpoint, which does not separate them), merger.
    """
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

    patch_embed = gemm(vs, 3 * temporal * patch * patch, hidden)

    units = vs // (merge * merge)
    side = int(round(units ** 0.5))
    window_units = int(v["window_size"]) // merge // patch
    windows_per_side = -(-side // window_units)          # ceil
    window_patches = (window_units * merge) ** 2
    windowed = [window_patches] * (windows_per_side * windows_per_side)
    full_layers = set(int(i) for i in v["fullatt_block_indexes"])

    qkv = 0
    o_plus_mlp = 0
    attention = 0
    for layer in range(depth):
        qkv += 3 * gemm(vs, hidden, hidden)              # q, k, v
        o_plus_mlp += gemm(vs, hidden, hidden)            # o
        o_plus_mlp += 3 * gemm(vs, hidden, inter)          # SwiGLU gate/up/down
        attention += _attention(
            [vs] if layer in full_layers else windowed, hidden)

    merged_dim = hidden * merge * merge
    merger = (gemm(merged, merged_dim, merged_dim)
              + gemm(merged, merged_dim, int(v["out_hidden_size"])))

    return {
        "patch_embed": patch_embed,
        "qkv_proj": qkv,
        "permute_qkv": 0,
        "rope": 0,
        "attention": attention,
        "unpermute+trim": 0,
        "o_proj+mlp": o_plus_mlp,
        "merger": merger,
    }


def vision_flops(cfg: dict, patches: int | None = None,
                 merged_tokens: int | None = None) -> int:
    """One image through the ViT tower: patch embed, 32 blocks, merger."""
    return sum(vision_flops_by_phase(cfg, patches, merged_tokens).values())


def audio_flops(cfg: dict, conv1_rows: int, encoder_states: int,
                chunk_states: Sequence[int], pooled_tokens: int) -> int:
    """One audio item: two convolutions, 32 blocks, the output projection.

    ``chunk_states`` is the post-convolution state count of each 200-mel-frame
    chunk.  Attention is confined to a chunk, so the chunks are the windows.

    No by-phase variant: --profile does not have an audio section (audio is
    never profiled for this model -- ``self._program_stage_profiles["audio"]``
    is unconditionally False), so there is no compiled checkpoint structure
    to line a breakdown up against.
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


def _lm_layer_breakdown(cfg: dict, tokens: int,
                        kv_lengths: Sequence[int]) -> dict[str, int]:
    """qkv/o/attention/mlp for ``tokens`` rows, keyed by the real checkpoint
    names _make_ckpt emits in qwen2.5_omni_7b_lm.py -- shared by prefill and
    decode, since this architecture (unlike gemma4_e2b) is uniform across
    layers: no sliding window, no per-layer MLP width change.
    """
    f = cfg["file_info"]
    hidden = int(f["hidden_size"])
    head_dim = int(f["head_dim"])
    kv_dim = int(f["num_kv_heads"]) * head_dim
    query_dim = int(f["num_kv_heads"]) * int(f["group_size"]) * head_dim
    inter = int(f["mlp_elements"])
    layers = int(f["num_layers"])
    t = int(tokens)

    return {
        "qkv": layers * (gemm(t, hidden, query_dim) + 2 * gemm(t, hidden, kv_dim)),
        "o": layers * gemm(t, query_dim, hidden),
        "mlp_gate_up": layers * (2 * gemm(t, hidden, inter)),
        "mlp_down": layers * gemm(t, inter, hidden),
        "attention": layers * (4 * query_dim
                               * sum(int(n) for n in kv_lengths)),
    }


def prefill_flops_by_phase(cfg: dict, seq_len: int) -> dict[str, int]:
    """Same total as prefill_flops, broken out by phase.

    Prefill's REAL multi-engine checkpoint structure is coarser than
    decode's -- confirmed from source (qwen2.5_omni_7b_lm.py's _emit_layer,
    the "else" branch active whenever a scheduler is supplied): gate, up,
    every down lane and the residual all run as ONE sharded region with no
    checkpoint between them, ending in a single ckpt(f"L{li}:mlp_proj", ...)
    that covers the WHOLE MLP -- unlike decode, whose gate/up and down sit
    on either side of separate "mlp_gate_up" and "mlp_proj" checkpoints (a
    real, source-verified difference, not a copy of decode's split -- an
    earlier version of this function assumed the two stages shared identical
    phase boundaries and was caught wrong by the real profile table missing
    an "mlp_gate_up" row entirely and "mlp_proj" coming out inflated 3.4x).
    Prefill also has no final_norm/mlp_norm/lm_head checkpoint (confirmed
    from source: all three ckpt(...) calls for those live inside the
    decode-only branch or _compile_decoder_impl), matching prefill_flops's
    own "no LM head here".
    """
    n = int(seq_len)
    layer = _lm_layer_breakdown(cfg, n, [n * (n + 1) // 2])
    return {
        "qkv_proj": layer["qkv"],
        "rope+cache": 0,
        "attention": layer["attention"],
        "attn_permute": 0,
        "o_proj": layer["o"],
        "mlp_proj": layer["mlp_gate_up"] + layer["mlp_down"],
    }


def prefill_flops(cfg: dict, seq_len: int) -> int:
    """Prefill fills the KV cache and stops -- there is no LM head here.

    Causal attention: row i attends i + 1 keys, so the KV lengths sum to
    n(n + 1)/2 rather than n^2.
    """
    return sum(prefill_flops_by_phase(cfg, seq_len).values())


def decode_step_flops_by_phase(cfg: dict, context_len: int) -> dict[str, int]:
    """Same total as decode_step_flops, broken out by phase, matching
    decode's real checkpoint structure exactly (qkv_proj, rope+cache,
    attention, attn_permute, o_proj, mlp_norm, mlp_gate_up, mlp_proj,
    final_norm, lm_head) -- decode's "if decode:" branch in _emit_layer DOES
    separate gate/up from down with its own checkpoints, unlike prefill's
    bundled one (see prefill_flops_by_phase).
    """
    layer = _lm_layer_breakdown(cfg, 1, [int(context_len)])
    return {
        "qkv_proj": layer["qkv"],
        "rope+cache": 0,
        "attention": layer["attention"],
        "attn_permute": 0,
        "o_proj": layer["o"],
        "mlp_norm": 0,
        "mlp_gate_up": layer["mlp_gate_up"],
        "mlp_proj": layer["mlp_down"],
        "final_norm": 0,
        "lm_head": gemm(1, int(cfg["file_info"]["hidden_size"]),
                        int(cfg["file_info"]["embedding_vocab"])),
    }


def decode_step_flops(cfg: dict, context_len: int) -> int:
    """One greedy step at ``context_len`` KV rows, including the LM head."""
    return sum(decode_step_flops_by_phase(cfg, context_len).values())


def decode_flops(cfg: dict, context_lengths: Sequence[int]) -> int:
    """Every step of a decode run, priced at the KV length it actually saw."""
    return sum(decode_step_flops(cfg, n) for n in context_lengths)
