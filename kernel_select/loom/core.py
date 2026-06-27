#!/usr/bin/env python3
"""loom.core — the auto-detect brain. Hardware-free, model-agnostic.

One file, three concerns, zero model-specific code:

    IR        Tensor / Op / Graph — the canonical data spine.
    DETECT    Graph -> repeated body + dims (topology) ; config -> archetype bit.
    WORKFLOW  Workflow base + BlockStack / Decode / Connector.

The whole thesis lives in find_repeated_body() + classify(): the repeated block
and its dimensions come from the GRAPH (identical for an encoder stack and a
decoder's traced forward); the archetype comes from ONE config bit (causal LM
with a KV cache -> decode, else block_stack). Everything else (M, which pointer
strides, weights-fixed) is DERIVED, never detected.

engine.py turns a Workflow into captured FPGA programs and runs them; run.py is
the single entry point. Nothing in this file imports torch or the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def pad64(n: int) -> int:
    """64-ALU floor: every hardware row width rounds up to a multiple of 64."""
    return ((n + 63) // 64) * 64


# ========================================================================== #
# IR
# ========================================================================== #
@dataclass
class Tensor:
    """A named value. Pre-lowering it is pure metadata; the harness stamps
    `.addr` at DRAM placement. One type covers SSA results, weights, and ports."""
    name: str
    shape: tuple
    dtype: str = "bf16"
    addr: Optional[int] = None
    role: str = "ssa"               # ssa | input | output | weight | const
    scale: Any = None

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def mn(self) -> tuple:
        """Hardware's 2-D (M, N) view: all but last dim fold into M, last is N."""
        if self.rank == 0:
            return (1, 1)
        m = 1
        for d in self.shape[:-1]:
            m *= d
        return (m, self.shape[-1])

    @property
    def n_pad(self) -> int:
        return pad64(self.shape[-1]) if self.rank else 64

    def __repr__(self) -> str:
        a = f"@{self.addr:#x}" if self.addr is not None else "@?"
        return f"Tensor({self.name} {list(self.shape)} {self.dtype} {self.role} {a})"


@dataclass
class Op:
    """One IR operation. `name` is a ue-dialect op the kernel registry knows.
    `operands`/`result` are SSA refs; `attrs` carries op knobs; `weights` names
    the bound parameter Tensors (filled by the frontend)."""
    name: str
    operands: list
    result: str
    out_shape: tuple
    attrs: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)

    VIEW = {"reshape", "permute", "select", "transpose", "view", "expand"}
    NORM = {"layer_norm", "rms_norm"}
    ATTN = {"flash_attn", "flash_attn_pf", "group_attn", "decode_attention"}
    GEMM = {"matmul", "qmatmul"}

    @property
    def is_view(self) -> bool:
        return self.name in self.VIEW

    @property
    def is_compute(self) -> bool:
        return not self.is_view

    @property
    def klass(self) -> str:
        if self.name in self.GEMM:
            return "gemm"
        if self.name in self.NORM:
            return "norm"
        if self.name in self.ATTN:
            return "attn"
        if self.name in self.VIEW:
            return "view"
        return self.name

    def __repr__(self) -> str:
        ops = ", ".join(self.operands)
        a = (" " + " ".join(f"{k}={v}" for k, v in self.attrs.items())) if self.attrs else ""
        return f"{self.result} = {self.name}({ops}){a} -> {list(self.out_shape)}"

    # --- ue-dialect text (this is what gets dumped to loom_dumps/) ---------- #
    def to_mlir(self, g: "Graph") -> str:
        def ty(s):
            t = g.tensors.get(s)
            return ("x".join(map(str, t.shape)) + "xbf16") if t else "?"
        ins = ", ".join(f"{s}: {ty(s)}" for s in self.operands)
        w = ("".join(f" {{{k}}}" for k in self.weights)) if self.weights else ""
        a = (" {" + ", ".join(f"{k}={v}" for k, v in self.attrs.items()) + "}") if self.attrs else ""
        return f"  {self.result} = ue.{self.name}({ins}){w}{a} : {ty(self.result)}"


class Graph:
    """Ordered ops + producer/consumer indices. Built once by the frontend; read
    by detect and workflow. Repeated-body machinery is pure topology."""

    def __init__(self, name: str = "graph"):
        self.name = name
        self.ops: list[Op] = []
        self.tensors: dict[str, Tensor] = {}
        self.inputs: list[str] = []
        self.outputs: list[str] = []

    def add_tensor(self, t: Tensor) -> Tensor:
        self.tensors[t.name] = t
        return t

    def add_op(self, op: Op) -> Op:
        self.ops.append(op)
        return op

    @property
    def producer(self) -> dict:
        return {op.result: op for op in self.ops}

    @property
    def consumers(self) -> dict:
        out: dict = {}
        for op in self.ops:
            for s in op.operands:
                out.setdefault(s, []).append(op)
        return out

    def compute_ops(self) -> list:
        return [o for o in self.ops if o.is_compute]

    def find_repeated_body(self) -> Optional["RepeatedBody"]:
        """Largest contiguous run of compute ops whose class-signature repeats
        back-to-back N>=2 times. That run IS the transformer block; N is depth.
        Brute force over periods — a block is tens of ops, cheap."""
        cops = self.compute_ops()
        n = len(cops)
        if n < 4:
            return None
        sig = [o.klass for o in cops]
        best = None
        for p in range(2, n // 2 + 1):
            i = 0
            while i + p <= n:
                base = sig[i:i + p]
                reps, j = 1, i + p
                while j + p <= n and sig[j:j + p] == base:
                    reps += 1
                    j += p
                if reps >= 2:
                    cand = (reps * p, p, i, reps)
                    if best is None or cand > best:
                        best = cand
                    i = j
                else:
                    i += 1
        if best is None:
            return None
        _, period, start, reps = best
        return RepeatedBody(self, cops[start:start + period], reps, start, period)

    def to_mlir(self) -> str:
        """Emit the ue-dialect text for the whole graph (the dump artifact)."""
        args = ", ".join(f"{s}: {'x'.join(map(str, self.tensors[s].shape))}xbf16"
                         for s in self.inputs if s in self.tensors)
        lines = [f"ue.func @{self.name}({args}) {{"]
        lines += [op.to_mlir(self) for op in self.ops]
        rets = ", ".join(self.outputs)
        lines.append(f"  ue.return {rets}")
        lines.append("}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.ops)

    def __repr__(self) -> str:
        return f"Graph({self.name}: {len(self.ops)} ops, {len(self.tensors)} tensors)"


@dataclass
class RepeatedBody:
    graph: Graph
    body: list
    depth: int
    start_compute_idx: int
    period: int

    def dims(self) -> dict:
        first = self.body[0]
        in_t = self.graph.tensors.get(first.operands[0]) if first.operands else None
        m = in_t.mn[0] if in_t else None
        hidden = in_t.shape[-1] if in_t else None
        gemms = [self.graph.tensors[o.result] for o in self.body
                 if o.klass == "gemm" and o.result in self.graph.tensors]
        widths = [t.shape[-1] for t in gemms if t is not None]
        ffn = max(widths) if widths else hidden
        attn = next((o for o in self.body if o.klass == "attn"), None)
        return {"depth": self.depth, "tokens": m, "hidden": hidden, "ffn": ffn,
                "n_gemm": len(gemms), "has_attn": attn is not None,
                "attn_attrs": attn.attrs if attn is not None else {}}

    def __repr__(self) -> str:
        d = self.dims()
        return (f"RepeatedBody(depth={self.depth} x {self.period} ops; "
                f"tokens={d['tokens']} hidden={d['hidden']} ffn={d['ffn']})")


# ========================================================================== #
# DETECT
# ========================================================================== #
BLOCK_STACK = "block_stack"
DECODE = "decode"
CONNECTOR = "connector"


@dataclass
class Archetype:
    """A PBI loop's shape. Two real ones, OPPOSITE address recurrences:
      block_stack : trip=depth, WEIGHTS stride, M=tokens.            [Swin/ViT]
      decode      : trip=dynamic, weights fixed, KV pointer strides, M=1. [LLM]
    connector is the degenerate no-loop glue case."""
    kind: str
    trip: object
    advancing: tuple
    weights_fixed: bool
    m_is_one: bool
    programs: tuple

    @staticmethod
    def block_stack(depth):
        return Archetype(BLOCK_STACK, depth, ("weights", "activation"),
                         False, False, ("forward",))

    @staticmethod
    def decode():
        return Archetype(DECODE, "dynamic", ("kv_cache",), True, True,
                         ("prefill", "decode"))

    @staticmethod
    def connector():
        return Archetype(CONNECTOR, 1, (), True, False, ("forward",))

    def __repr__(self):
        return (f"Archetype({self.kind} trip={self.trip} "
                f"advancing={'+'.join(self.advancing) or '-'} "
                f"weights_fixed={self.weights_fixed} M=1:{self.m_is_one})")


@dataclass
class Region:
    name: str
    role: str                  # lm | vision_encoder | audio_encoder | connector
    module: Any
    config: Any
    is_backbone: bool = False


def is_causal_lm(cfg) -> bool:
    """The single discriminator. Read off config, NOT the graph: a decoder's
    traced forward looks exactly like an encoder block stack."""
    if cfg is None:
        return False
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if "swin" in mt or hasattr(cfg, "depths") or "vit" in mt or "encoder" in mt:
        return False
    archs = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]
    if any("forcausallm" in a or "forconditionalgeneration" in a for a in archs):
        return True
    if getattr(cfg, "use_cache", False) and getattr(cfg, "is_decoder", False):
        return True
    if hasattr(cfg, "num_hidden_layers") and (
            hasattr(cfg, "num_key_value_heads") or getattr(cfg, "use_cache", False)):
        return True
    return False


def is_block_stack_cfg(cfg) -> bool:
    """A non-causal repeated-block encoder. Config carries the structure (depths /
    a layer count); we trust it over graph topology because real encoders (Swin)
    have alternating-shift blocks + inter-stage merges that don't tile cleanly."""
    if cfg is None:
        return False
    mt = (getattr(cfg, "model_type", "") or "").lower()
    if hasattr(cfg, "depths"):
        return True
    if ("vit" in mt or "swin" in mt or "encoder" in mt) and hasattr(cfg, "num_hidden_layers"):
        return True
    return False


def dims_from_config(cfg, role: str) -> dict:
    """Pull block-stack / decode dimensions straight off the HF config — the same
    facts plan_swin/plan_decoder used. Returns {} when config is uninformative
    (the graph body then supplies dims)."""
    if cfg is None:
        return {}
    g = lambda k, d=None: getattr(cfg, k, d)
    if hasattr(cfg, "depths"):                      # Swin-style multi-stage encoder
        depths = list(g("depths"))
        heads = list(g("num_heads", [])) or [None] * len(depths)
        embed = int(g("embed_dim", 0))
        img, patch = int(g("image_size", 0)), int(g("patch_size", 1))
        res = (img // patch) if patch else 0
        stages = []
        dim, r = embed, res
        for i, d in enumerate(depths):
            stages.append({"index": i, "depth": d, "dim": dim,
                           "heads": heads[i], "resolution": r, "tokens": r * r})
            r, dim = r // 2, dim * 2
        return {"depth": sum(depths), "stages": stages, "hidden": embed,
                "tokens": res * res, "ffn": embed * 4, "n_stages": len(depths),
                "has_attn": True}
    if hasattr(cfg, "num_hidden_layers"):           # plain transformer / decoder
        nq = int(g("num_attention_heads", g("n_head", 0)) or 0)
        h = int(g("hidden_size", g("n_embd", 0)) or 0)
        return {"depth": int(g("num_hidden_layers")), "hidden": h,
                "tokens": 1 if role == "lm" else None,
                "ffn": int(g("intermediate_size", h * 4) or h * 4),
                "heads": nq, "kv_heads": int(g("num_key_value_heads", nq) or nq),
                "head_dim": int(g("head_dim", (h // nq) if nq else 0) or 0),
                "has_attn": True}
    return {}


def classify(region: Region, body: Optional[RepeatedBody]) -> Archetype:
    """Config-first: the archetype bit and the depth both come from config when it
    carries them (robust); graph body is the fallback for configs that don't."""
    if region.role == "connector":
        return Archetype.connector()
    if is_causal_lm(region.config):
        return Archetype.decode()
    # a vision/audio encoder is a non-causal block stack even when its config
    # counts layers with num_hidden_layers (ViT/SigLIP) rather than depths (Swin)
    if region.role in ("vision_encoder", "audio_encoder") or is_block_stack_cfg(region.config):
        d = dims_from_config(region.config, region.role)
        return Archetype.block_stack(d.get("depth", body.depth if body else 1))
    if body is not None:
        return Archetype.block_stack(body.depth)
    return Archetype.connector()


_VISION = ("vision", "image", "visual", "patch", "vit", "swin", "clip", "moonvit", "siglip")
_AUDIO = ("audio", "speech", "wav", "mel", "conformer", "whisper")
_CONNECT = ("connector", "projector", "mm_proj", "multi_modal", "merger")
_LM = ("text_model", "language_model", "decoder", "llm", "causallm", "llama",
       "qwen", "gemma", "mistral", "gpt")


def _role_of(name, module):
    blob = name.lower() + " " + type(module).__name__.lower()
    if any(h in blob for h in _CONNECT):
        return "connector"
    if any(h in blob for h in _AUDIO):
        return "audio_encoder"
    if any(h in blob for h in _VISION):
        return "vision_encoder"
    if any(h in blob for h in _LM):
        return "lm"
    return None


def _find_backbone(model):
    import torch.nn as nn
    best = None
    for qual, mod in model.named_modules():
        for _, child in mod.named_children():
            if isinstance(child, nn.ModuleList) and len(child) >= 2:
                cand = (len(child), qual or "model", mod)
                if best is None or cand[0] > best[0]:
                    best = cand
    return (best[1], best[2]) if best else None


def partition(model, config) -> list:
    """Model -> ordered execution regions. Single backbone for a plain model;
    encoder(s)+connector+LM for a VLM. Pure structure walk, no model names.

    A named encoder region (e.g. `swin`) already CONTAINS its block ModuleList, so
    we must not also emit that inner ModuleList as a separate 'backbone' — we skip
    any backbone whose owner is a submodule of an already-claimed region."""
    regions = []

    def sub(role):
        """Resolve the per-region HF sub-config. Connector has no decoder/encoder
        config — return None so it doesn't borrow a stack's dimensions."""
        key = {"vision_encoder": "vision_config", "audio_encoder": "audio_config",
               "lm": "text_config"}.get(role)
        if key is None:
            return None if role == "connector" else config
        return getattr(config, key, None) or config

    claimed_ids = set()

    def walk(module, prefix, depth):
        """Recurse into role-less containers so nested regions are found whatever
        the wrapper depth (e.g. ForConditionalGeneration.model.{vision_model,
        connector,text_model}). A role-bearing module is claimed, not descended."""
        if depth > 4:
            return
        for name, child in module.named_children():
            qual = f"{prefix}.{name}" if prefix else name
            role = _role_of(name, child)
            if role:
                regions.append(Region(qual, role, child, sub(role)))
                claimed_ids.update(id(m) for m in child.modules())
            else:
                walk(child, qual, depth + 1)

    walk(model, "", 0)

    bb = _find_backbone(model)
    if bb is not None and id(bb[1]) not in claimed_ids:
        regions.append(Region(bb[0], "lm", bb[1],
                              getattr(config, "text_config", None) or config, True))
    elif not regions:
        regions.append(Region("model", "lm", model, config, True))

    rank = {"vision_encoder": 0, "audio_encoder": 0, "connector": 1, "lm": 2}
    return sorted(regions, key=lambda r: rank.get(r.role, 3))


# ========================================================================== #
# WORKLOAD — the flat, numbered forward-pass schedule
# ========================================================================== #
# A forward pass is an ORDERED LIST of workloads, each a primitive execution
# stage. The compiler builds novel forward passes it has never seen by composing
# these primitives — an unseen model is just a new ordering of the same kinds.
#
#   encode    a block stack (vision/audio/non-causal encoder) — weights stride
#   connect   projection glue between regions (vision -> text embedding space)
#   prefill   causal decoder over the full prompt (M=tokens, populates KV)
#   decode    causal decoder M=1 per-token loop (KV pointer advances)
#
# An archetype EXPANDS into workloads: block_stack -> [encode]; connector ->
# [connect]; decode -> [prefill, decode]. Prefill and decode are PEER workloads,
# not two programs hidden in one archetype — that is what makes the schedule
# composable. Activations thread workload[i] -> workload[i+1] via an io dict.
ENCODE = "encode"
CONNECT = "connect"
PREFILL = "prefill"
DECODE_W = "decode"      # workload kind (distinct from the DECODE archetype const)

# archetype.kind -> the ordered workload kinds it expands into
_EXPAND = {
    BLOCK_STACK: [ENCODE],
    CONNECTOR:   [CONNECT],
    DECODE:      [PREFILL, DECODE_W],
}


@dataclass
class Workload:
    """One numbered stage of a forward pass. Carries everything execution needs:
    the source region, the archetype it came from, the dims, and the optional
    lowered graph. `index` is its position in the global schedule (w0, w1, ...)."""
    index: int
    kind: str                 # encode | connect | prefill | decode
    region: Region
    archetype: Archetype
    dims: dict
    graph: Any = None
    body: Any = None

    @property
    def name(self):
        return f"w{self.index}:{self.kind}"

    def report(self):
        d = self.dims
        head = f"{self.name:<12} [{self.region.name}]  {self.archetype.kind}"
        if not d:
            return head + "  (glue, no repeated body)"
        line = (f"{head}\n   dims: depth={d.get('depth')} tokens={d.get('tokens')} "
                f"hidden={d.get('hidden')} ffn={d.get('ffn')} "
                f"heads={d.get('heads')} kv_heads={d.get('kv_heads')} "
                f"head_dim={d.get('head_dim')}")
        if d.get("stages"):
            for s in d["stages"]:
                line += (f"\n     stage{s['index']}: depth={s['depth']} dim={s['dim']} "
                         f"heads={s['heads']} res={s['resolution']} tokens={s['tokens']}")
        return line

    def __repr__(self):
        return f"<Workload {self.name} {self.region.name}>"


def schedule(model, config, *, lower=None) -> list:
    """Model -> ordered, numbered [Workload]: the forward-pass schedule. Detect
    regions, classify each archetype, then EXPAND each into its workload kinds and
    number them 0..N in dataflow order. This list IS the program the executor
    walks; composing a new model = a new such list, no new execution code."""
    workloads = []
    i = 0
    for r in partition(model, config):
        graph = lower(r.module, r.config) if lower is not None else None
        body = graph.find_repeated_body() if graph is not None else None
        arch = classify(r, body)
        dims = dims_from_config(r.config, r.role) or (body.dims() if body else {})
        for kind in _EXPAND.get(arch.kind, [ENCODE]):
            workloads.append(Workload(i, kind, r, arch, dims, graph, body))
            i += 1
    return workloads
