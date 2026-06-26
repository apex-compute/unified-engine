#!/usr/bin/env python3
"""loom_plan.py — Layer A: workload structure -> program plan.

The torch.export FX graph (Layer B) is pure forward-pass dataflow; it CANNOT
express the program structure (single forward vs prefill+decode loop) or the PBI
policy. That comes from HERE: classify the workload, then emit a ProgramPlan that
says which programs exist and what drives PBI in each.

PBI falls out of the plan automatically:
    PBI(op) = (the op's stage has a loop_reg)  ×  (op is PBI-capable in hlo.py)

For an encoder (Swin): ONE forward program, with one loop register PER STAGE
(stages have different token counts), each primed to that stage's token count at
execute. matmul/layer_norm/eltwise -> PBI on the stage register; attention /
windowing -> legacy (not PBI-capable). No decode loop.
"""
from dataclasses import dataclass, field


def _pad64(n):
    return ((n + 63) // 64) * 64


@dataclass
class AttentionSpec:
    """Parameterized attention — the SAME engine for vision-windowed AND LLM-causal,
    different knobs. Layer A derives this from config; the emitter picks the flash
    kernel + staging + masking from it. The knob set is grounded in the repo's
    models (gemma3/4, llama, qwen2.5/3, swin, smolvlm2, parakeet, ...):
      - GQA group ∈ {1,2,3,4,6,8}    (num_q_heads // num_kv_heads)
      - head_dim ∈ {32,64,128,256,512}, seq ∈ {144 window .. 4096}
      - rope ∈ {none, global, local, mixed, mrope, partial}
      - kv_cache True for every decoder, False for every encoder
      - sliding_window > 0 for gemma4/qwen3.5 windowed-causal layers."""
    kind: str            # "causal" | "global" | "windowed"
    num_q_heads: int
    num_kv_heads: int    # == num_q_heads unless GQA
    head_dim: int
    seq_len: int         # full seq (causal/global) OR window_area (windowed)
    batch: int           # 1 (causal/global) OR num_windows (windowed)
    bias: str            # "causal" | "none" | "relative_position"
    # --- LLM-path knobs (defaults keep the vision-encoder call sites unchanged) ---
    rope: str = "none"          # "none"|"global"|"local"|"mixed"|"mrope"|"partial"
    uses_kv_cache: bool = False # True for decoders (decode loop appends K/V)
    sliding_window: int = 0     # 0 = full attention; >0 = causal sliding window
    @property
    def group(self):                 # GQA group size (1 = MHA, >1 = GQA, ==q = MQA)
        return self.num_q_heads // self.num_kv_heads
    @property
    def is_gqa(self):
        return self.num_kv_heads < self.num_q_heads
    @property
    def head_dim_pad(self):          # 64-ALU floor
        return _pad64(self.head_dim)
    @property
    def seq_pad(self):
        return _pad64(self.seq_len)
    @property
    def q_prescale(self):            # corrects the padded-head_dim softmax scale
        import math
        return math.sqrt(self.head_dim_pad) / math.sqrt(self.head_dim)

    def kernel_for(self, phase="prefill"):
        """Pick the hardware flash kernel from the knobs + execution phase.
        This is the single dispatch point that lets ONE emitter serve vision
        and LLM: windowed->batched, decode->group(GQA), prefill->flash core."""
        if self.kind == "windowed":          # vision encoder: batched per-window
            return "flash_attention_batched_pbi"
        if phase == "decode":                # LLM decode: M=1, KV-cache read, GQA fan-out
            return "decoder_group_attention_core"
        return "flash_attention_core"        # causal/global prefill

    def __repr__(self):
        extra = []
        if self.rope != "none": extra.append(f"rope={self.rope}")
        if self.uses_kv_cache: extra.append("kv$")
        if self.sliding_window: extra.append(f"swin={self.sliding_window}")
        x = (" " + " ".join(extra)) if extra else ""
        return (f"AttentionSpec({self.kind} heads={self.num_q_heads}/{self.num_kv_heads} "
                f"hd={self.head_dim}->{self.head_dim_pad} seq={self.seq_len}->{self.seq_pad} "
                f"batch={self.batch} bias={self.bias}{x})")


@dataclass
class LoopArchetype:
    """A PBI loop's *shape* — the thing the reroller must recognize. There are
    exactly two on this hardware, with OPPOSITE address recurrences:

      block_stack (vision encoder): one re-used body, trip = stage depth.
          weights ADVANCE per iter (+stride), activation chains head->tail,
          M = token count (large). What Swin does.

      decode (LLM): one re-used body, trip = generated tokens (dynamic).
          weights are FIXED (same layers every token), the KV pointer ADVANCES
          (+1 token / iter, seq grows), M = 1. The original PBI decode case.

    Same loop_reg mechanism; the analyzer distinguishes them by WHICH pointer
    strides. If you only know block_stack you can never re-roll a decoder."""
    kind: str               # "block_stack" | "decode"
    trip_reg: str           # GPR primed to the trip count
    trip: object            # int (depth) for block_stack; "dynamic" for decode
    advancing: tuple        # which pointers stride per iter
    weights_fixed: bool     # decode: True ; block_stack: False
    m_is_one: bool          # decode: True (single token) ; block_stack: False

    @staticmethod
    def block_stack(trip_reg, depth):
        return LoopArchetype("block_stack", trip_reg, depth,
                             advancing=("weights", "activation"),
                             weights_fixed=False, m_is_one=False)

    @staticmethod
    def decode(trip_reg):
        return LoopArchetype("decode", trip_reg, "dynamic",
                             advancing=("kv_cache",),
                             weights_fixed=True, m_is_one=True)

    def __repr__(self):
        return (f"LoopArchetype({self.kind} trip={self.trip} via {self.trip_reg}; "
                f"advancing={'+'.join(self.advancing)}; "
                f"weights_fixed={self.weights_fixed} M=1:{self.m_is_one})")


@dataclass
class StagePlan:
    """One Swin stage: a run of `depth` blocks at a fixed resolution/dim."""
    index: int
    tokens: int          # M for the row-wise PBI ops (num_windows * window_area == H*W)
    dim: int             # channel width (N)
    heads: int
    depth: int           # number of transformer blocks in this stage
    resolution: int      # token-grid side length (H == W)
    loop_reg: str        # name of the GPR primed to `tokens` at execute
    attn: "AttentionSpec" = None        # parameterized attention config for this stage
    archetype: "LoopArchetype" = None   # block_stack (vision) | decode (LLM)


@dataclass
class ProgramPlan:
    """The Layer-A output consumed by the walker (Layer B)."""
    name: str
    kind: str                                  # "encoder" | "decoder"
    programs: list[str]                        # e.g. ["forward"] or ["prefill","decode"]
    stages: list[StagePlan] = field(default_factory=list)
    patch: dict = field(default_factory=dict)  # patch-embed: in_ch, patch, out_dim, tokens
    binds: dict = field(default_factory=dict)  # loop_reg name -> primed value (token count)

    def report(self) -> str:
        L = [f"ProgramPlan({self.name}) kind={self.kind} programs={self.programs}"]
        p = self.patch
        if p:
            L.append(f"  patch-embed: {p.get('in_ch')}ch {p.get('image')}px /{p.get('patch')} "
                     f"-> {p.get('tokens')} tokens x {p.get('out_dim')}d")
        for s in self.stages:
            L.append(f"  stage {s.index}: res {s.resolution}x{s.resolution}  "
                     f"tokens={s.tokens:5d}  dim={s.dim:4d}  heads={s.heads:2d}  "
                     f"depth={s.depth}  PBI-reg={s.loop_reg}(={s.tokens})")
            L.append(f"           attn: {s.attn}")
            L.append(f"           loop: {s.archetype}")
            L.append(f"           kernel: prefill={s.attn.kernel_for('prefill')}  "
                     f"decode={s.attn.kernel_for('decode')}")
        nblk = sum(s.depth for s in self.stages)
        L.append(f"  total blocks={nblk}  execute binds={self.binds}")
        return "\n".join(L)


def plan_swin(cfg) -> ProgramPlan:
    """Derive the encoder program plan from a Swin HF config."""
    g = (lambda k, d=None: getattr(cfg, k, d))
    depths = list(g("depths"))
    heads = list(g("num_heads"))
    embed = int(g("embed_dim"))
    img = int(g("image_size"))
    patch = int(g("patch_size"))
    in_ch = int(g("num_channels", 3))

    res = img // patch                  # 224/4 = 56
    tokens = res * res                  # 3136
    dim = embed                         # 96

    plan = ProgramPlan(name="swin", kind="encoder", programs=["forward"])
    plan.patch = {"in_ch": in_ch, "image": img, "patch": patch,
                  "out_dim": dim, "tokens": tokens}

    ws = int(g("window_size"))
    for i, (depth, nh) in enumerate(zip(depths, heads)):
        reg = f"tok_s{i}"               # one PBI register per stage
        win_side = min(ws, res)         # last stage: window == full grid
        num_windows = (res // win_side) ** 2
        attn = AttentionSpec(
            kind="windowed", num_q_heads=nh, num_kv_heads=nh,
            head_dim=dim // nh, seq_len=win_side * win_side,
            batch=num_windows, bias="relative_position")
        plan.stages.append(StagePlan(
            index=i, tokens=tokens, dim=dim, heads=nh,
            depth=depth, resolution=res, loop_reg=reg, attn=attn,
            archetype=LoopArchetype.block_stack(reg, depth)))   # weights stride, M=tokens
        plan.binds[reg] = tokens
        # patch-merging halves resolution, doubles channels, for the next stage
        if i < len(depths) - 1:
            res //= 2
            tokens = res * res
            dim *= 2
    return plan


def plan_decoder(cfg) -> ProgramPlan:
    """Derive an LLM program plan: prefill + a per-token decode loop. The decode
    loop is the OTHER PBI archetype — weights fixed, KV pointer advances, M=1.
    Grounded in the repo's decoders (gemma3/4, llama3.2, qwen2.5/3, smolvlm2)."""
    g = (lambda *ks, d=None: next((getattr(cfg, k) for k in ks if hasattr(cfg, k)), d))
    n_layer = int(g("num_hidden_layers", "n_layer", d=0))
    n_q = int(g("num_attention_heads", "n_head", d=0))
    n_kv = int(g("num_key_value_heads", d=n_q))           # == n_q unless GQA
    hidden = int(g("hidden_size", "n_embd", d=0))
    head_dim = int(g("head_dim", d=(hidden // n_q if n_q else 0)))
    seq = int(g("max_position_embeddings", "n_positions", d=4096))
    swin = int(g("sliding_window", d=0) or 0)
    rope = "none" if g("model_type", d="") == "gpt2" else "global"

    attn = AttentionSpec(
        kind="causal", num_q_heads=n_q, num_kv_heads=n_kv, head_dim=head_dim,
        seq_len=seq, batch=1, bias="causal",
        rope=rope, uses_kv_cache=True, sliding_window=swin)

    plan = ProgramPlan(name=g("model_type", d="decoder"), kind="decoder",
                       programs=["prefill", "decode"])
    reg = "kv_pos"                                          # GPR = current KV length
    # one StagePlan standing for the repeated decoder layer (depth = n_layer),
    # carrying the DECODE archetype (the prefill reuses the same body, M=prompt).
    plan.stages.append(StagePlan(
        index=0, tokens=1, dim=hidden, heads=n_q, depth=n_layer,
        resolution=0, loop_reg=reg, attn=attn,
        archetype=LoopArchetype.decode(reg)))
    plan.binds[reg] = 0                                     # primed per decode step
    return plan


def plan_for(model_id_or_cfg) -> ProgramPlan:
    """Layer-A entry: classify + emit a ProgramPlan. Encoder path only for now;
    decoder (prefill + decode-loop) is the later branch that reuses k_select."""
    cfg = model_id_or_cfg
    if isinstance(cfg, str):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(cfg)
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if "swin" in mt or hasattr(cfg, "depths"):
        return plan_swin(cfg)                              # vision: block_stack
    if hasattr(cfg, "num_hidden_layers") or hasattr(cfg, "n_layer"):
        return plan_decoder(cfg)                           # LLM: prefill + decode
    raise NotImplementedError(
        f"Layer A: no program-plan rule for model_type={mt!r} yet.")


if __name__ == "__main__":
    import sys
    mid = sys.argv[1] if len(sys.argv) > 1 else "microsoft/swin-tiny-patch4-window7-224"
    print(f"[loom_plan] classifying {mid}\n")
    print(plan_for(mid).report())
