"""workload_detect.py — Auto-detect workload boundaries from HuggingFace models.

Strategy: module-structure inspection to find modality encoders, connectors,
and LM backbone. Returns WorkloadSpec objects feedable into the WorkflowRunner.

Detection heuristics
---------------------
1. Walk top-level named_children().
2. Classify by class/attribute name patterns:
   - vision_encoder : "VisionTransformer", "ViT", "Swin", vision_tower, etc.
   - audio_encoder  : "AudioEncoder", "Whisper", audio_tower, etc.
   - connector      : "Projector", "Connector", multi_modal_projector, etc.
   - lm_root        : module containing a ModuleList of repeated DecoderLayer/Block
3. The LM workload gets is_backbone=True (handles prefill + decode).
4. Everything else is absorbed into the nearest workload boundary.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class PortSpec:
    """Shape-free port descriptor the detector emits."""
    name: str
    role: str = "input"            # input | output | inout
    shape: Optional[tuple] = None  # None = shape unknown; caller fills
    dtype: str = "bf16"


@dataclass
class WorkloadSpec:
    """Auto-detected workload definition. Consumed by WorkflowRunner wiring."""
    name: str
    type: str                      # lm | vision_encoder | audio_encoder | connector
    module: nn.Module = field(repr=False)
    module_path: str = ""          # dotted attribute path from model root
    input_ports: list[PortSpec] = field(default_factory=list)
    output_ports: list[PortSpec] = field(default_factory=list)
    num_layers: int = 0            # repeated block count
    is_backbone: bool = False      # True for LM (hosts prefill+decode)
    hidden_size: int = 0           # LM hidden dim (from config)


# =============================================================================
# Internal helpers
# =============================================================================

def _norm(val: str) -> str:
    """Lowercase, strip underscores and hyphens (casefolded match)."""
    return val.lower().replace("_", "").replace("-", "")


def _has_any(haystack: str, *needles: str) -> bool:
    hl = _norm(haystack)
    return any(_norm(n) in hl for n in needles)




def _classify(name: str, mod: nn.Module) -> Optional[str]:
    """Return workload type for a (name, module) pair, or None if internal."""
    cls = type(mod).__name__
    nl = name.lower().replace("_", "").replace("-", "")

    # ── Modality encoders ──────────────────────────────────────────────
    if _has_any(cls, "vision", "vit", "swin"):
        return "vision_encoder"
    if _has_any(name, "vision_tower", "vision_encoder", "vit", "swin"):
        return "vision_encoder"
    if _has_any(cls, "audio", "whisper"):
        return "audio_encoder"
    if _has_any(name, "audio_tower", "audio_encoder"):
        return "audio_encoder"

    # ── Connector ───────────────────────────────────────────────────────
    if _has_any(cls, "projector", "connector", "adapter", "merger"):
        return "connector"
    if _has_any(name, "projector", "connector", "adapter", "merger",
                "multimodalprojector", "mmprojector"):
        return "connector"

    # ── LM backbone (ModuleList of DecoderLayer/Block) ─────────────────
    if isinstance(mod, nn.ModuleList) and len(mod) > 1:
        child = type(mod[0]).__name__
        if _has_any(child, "block", "layer", "decoder"):
            return "lm_layers"
    for _, child in mod.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 1:
            if _has_any(type(child[0]).__name__, "block", "layer", "decoder"):
                return "lm_root"

    # Named container that wraps the LM
    if nl in ("model", "transformer", "languagemodel", "backbone"):
        return "lm_root"

    return None


def _count_layers(mod: nn.Module) -> int:
    """Find the first ModuleList of repeated blocks and return its size."""
    for _, child in mod.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 1:
            cls = type(child[0]).__name__
            if _has_any(cls, "block", "layer", "decoder"):
                return len(child)
        if isinstance(child, nn.Module):
            r = _count_layers(child)
            if r > 0:
                return r
    return 0


def _find_lm_root(model: nn.Module) -> Optional[nn.Module]:
    """Walk top-level children to find the module that contains the layers list."""
    # Direct match (ForCausalLM wrapper itself contains layers)
    if _count_layers(model) > 0:
        return model
    # One level deep
    for name, child in model.named_children():
        if _count_layers(child) > 0:
            return child
    # Two levels deep (e.g., model.model.layers)
    for _, child in model.named_children():
        if isinstance(child, nn.Module):
            for _, c2 in child.named_children():
                if _count_layers(c2) > 0:
                    return child
    return None


# =============================================================================
# Public API
# =============================================================================

def detect_workloads(model: nn.Module, config: dict = None) -> list[WorkloadSpec]:
    """Auto-detect workloads from a HuggingFace model.

    Args:
        model: HF model (or any nn.Module with named children).
        config: HF config dict (optional — used for hidden_size and shape inference).

    Returns:
        List of WorkloadSpec in forward-execution order.
    """
    config = config or {}
    specs: list[WorkloadSpec] = []
    used: set[str] = set()

    # ── Pass 1: modality encoders + connector ───────────────────────────
    encoders: list[tuple[str, nn.Module, str]] = []
    connector: Optional[tuple[str, nn.Module]] = None

    for name, child in model.named_children():
        if name.startswith("_"):
            continue
        wl_type = _classify(name, child)
        if wl_type == "vision_encoder":
            encoders.append((name, child, "vision_encoder"))
            used.add(name)
        elif wl_type == "audio_encoder":
            encoders.append((name, child, "audio_encoder"))
            used.add(name)
        elif wl_type == "connector":
            connector = (name, child)
            used.add(name)

    for name, mod, wl_type in encoders:
        is_vision = wl_type == "vision_encoder"
        wl_name = "vision_encoder" if is_vision else "audio_encoder"
        inp_name = "pixel_values" if is_vision else "audio_values"
        out_name = "image_embeds" if is_vision else "audio_embeds"
        specs.append(WorkloadSpec(
            name=wl_name,
            type=wl_type,
            module=mod,
            module_path=name,
            input_ports=[PortSpec(inp_name, role="input")],
            output_ports=[PortSpec(out_name, role="output")],
            num_layers=_count_layers(mod),
        ))

    if connector is not None:
        inp = specs[-1].output_ports[0].name if specs else "image_embeds"
        specs.append(WorkloadSpec(
            name="connector",
            type="connector",
            module=connector[1],
            module_path=connector[0],
            input_ports=[PortSpec(inp, role="input")],
            output_ports=[PortSpec("hidden_states", role="output")],
        ))

    # ── Pass 2: LM backbone ─────────────────────────────────────────────
    lm_mod = None
    lm_path = ""
    for name, child in model.named_children():
        if name in used:
            continue
        if _classify(name, child) in ("lm_root", "lm_layers"):
            lm_mod, lm_path = child, name
            used.add(name)
            break

    if lm_mod is None:
        lm_mod = _find_lm_root(model)
        lm_path = ""

    if lm_mod is None and "ForCausalLM" in type(model).__name__:
        lm_mod, lm_path = model, ""

    hidden = (config.get("hidden_size") or config.get("d_model")
              or config.get("n_embd", 0))
    if lm_mod is not None:
        out_shape = ("batch", "seq_len", hidden) if hidden else None
        specs.append(WorkloadSpec(
            name="lm",
            type="lm",
            module=lm_mod,
            module_path=lm_path,
            input_ports=[
                PortSpec("input_ids", role="input", dtype="int64"),
                PortSpec("attention_mask", role="input", dtype="int64"),
            ],
            output_ports=[PortSpec("logits", role="output", shape=out_shape)],
            num_layers=_count_layers(lm_mod),
            is_backbone=True,
            hidden_size=hidden,
        ))

    # ── Pass 3: fallback — whole model as LM ────────────────────────────
    if not specs:
        specs.append(WorkloadSpec(
            name="lm", type="lm", module=model,
            input_ports=[PortSpec("input_ids", role="input", dtype="int64")],
            output_ports=[PortSpec("logits", role="output")],
            num_layers=_count_layers(model),
            is_backbone=True,
        ))

    return specs


def apply_specs(specs: list[WorkloadSpec], runner) -> dict[str, Any]:
    """Convert WorkloadSpec list into WorkflowRunner Workload objects.

    Each spec becomes a Workload with programs named ``<wl.name>_prefill``
    and ``<wl.name>_decode`` (for backbone / LM workloads) or ``<wl.name>``
    (for modality encoders / connectors).  The caller fills in the actual
    compiled programs and tensor handles.

    Returns a dict mapping spec name -> (Workload, WorkloadSpec) for the
    caller to complete (bind GPRs, assign tensors to ports).
    """
    from ir_harness import Workload

    out = {}
    for spec in specs:
        wl = runner.workload(spec.name)
        if spec.is_backbone:
            wl.add(f"{spec.name}_prefill", f"{spec.name}_decode")
        else:
            wl.add(spec.name)

        for p in spec.input_ports:
            wl.input(p.name)
        for p in spec.output_ports:
            wl.output(p.name)

        out[spec.name] = (wl, spec)
    runner.wire()
    return out


def detect_merger_pattern(model) -> dict:
    """Auto-detect how vision features merge into the text decoder.

    Returns a dict with keys:
      ``pattern`` — ``"token_replacement"`` | ``"concat"`` | ``"cross_attention"``
      ``image_token_id`` — the special token replaced with image features (or None)
      ``merger_fn`` — the method/module that does the merge (or None)
    """
    import inspect
    cfg = model.config
    mm = getattr(model, "model", model)
    tcfg = getattr(cfg, "text_config", cfg)

    # ── Config signals ──
    image_token_id = getattr(cfg, "image_token_id", None)
    has_cross_attn = hasattr(tcfg, "cross_attention_layers") and tcfg.cross_attention_layers

    # ── Look for a merger method ──
    merger = None
    for name in ("inputs_merger", "merge_inputs", "interleave", "input_merger"):
        obj = getattr(mm, name, None)
        if obj is not None and isinstance(obj, type(lambda: 0)):
            merger = obj
            break

    # ── Classify ──
    if merger is not None:
        src = inspect.getsource(merger)
        if "torch.where" in src or "index_put" in src or "scatter" in src:
            pattern = "token_replacement"
        elif "torch.cat" in src:
            pattern = "concat"
        else:
            pattern = "token_replacement" if image_token_id else "concat"
    elif has_cross_attn:
        pattern = "cross_attention"
    elif image_token_id:
        pattern = "token_replacement"
    else:
        pattern = "concat"

    return {"pattern": pattern, "image_token_id": image_token_id, "merger_fn": merger}


def summarize(specs: list[WorkloadSpec]) -> str:
    """One-line per workload summary for display."""
    lines = []
    for s in specs:
        mod = type(s.module).__name__
        in_p = ", ".join(f"{p.name}:{p.role}" for p in s.input_ports)
        out_p = ", ".join(f"{p.name}:{p.role}" for p in s.output_ports)
        layers = f" {s.num_layers} layers" if s.num_layers else ""
        bb = " [backbone]" if s.is_backbone else ""
        lines.append(f"  {s.name:20s}  {mod:40s}  in=[{in_p}]  out=[{out_p}]{layers}{bb}")
    return "\n".join(lines)
