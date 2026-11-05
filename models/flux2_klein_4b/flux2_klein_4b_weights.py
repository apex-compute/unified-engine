#!/usr/bin/env python3
"""Weight export for the FLUX.2 [klein] 4B FPGA port.

Builds three device images from the HF checkpoint (M1 "model conversion"):

  params_te.bin  / params_te.json   Qwen3-4B text encoder, 27 layers, IF4
                                    projections + BF16 norms. One fixed
                                    per-layer region layout (layer stride) so
                                    the compiler addresses layer i as
                                    ``base + i * stride + region_offset``.
  embed_tokens.bin                  Qwen3 token embedding (BF16, host-side gather).
  params_dit.bin / params_dit.json  FLUX.2 transformer: per-double-block and
                                    per-single-block regions (IF4) plus the
                                    BF16 non-layer tensors (embedders, timestep
                                    MLP, shared modulation, output head).
  vae_weights.pt                    AutoencoderKLFlux2 decoder: IF8 conv codes +
                                    signed per-(oc,tap,64-ch) scales + biases,
                                    GroupNorm affines, BN latent statistics.

RoPE convention: FLUX.2 rotates *interleaved* pairs (2j, 2j+1). The engine's
RoPE core is HF rotate-half, so the Q/K projection rows of every head (and the
matching q/k RMSNorm gammas) are permuted to even-first order here; the
rotate-half table is then built from the pair frequencies (see
``flux2_klein_4b_ref.rotate_half_permutation``). Attention is invariant to the
common permutation of q and k, so nothing else changes.

All byte formats come from ``quant_lib`` (shared with every other model):
IF4 data = (N, K/2) nibbles (low nibble = even k), scale = (N, K/64) BF16,
negative scale = INT4 block, positive = FP4 block.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import quant_lib  # noqa: E402
import flux2_klein_4b_ref as ref  # noqa: E402

ALIGN = 4096  # region alignment inside a bin (keeps every DRAM address page-aligned)


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def _bf16_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().to(torch.bfloat16).view(torch.int16).numpy().tobytes()


class _BinBuilder:
    """Append-only region packer with a JSON manifest."""

    def __init__(self):
        self.buf = bytearray()
        self.regions: dict[str, dict] = {}

    def add(self, name: str, data: bytes, kind: str, shape) -> int:
        assert name not in self.regions, f"duplicate region {name}"
        off = _align(len(self.buf))
        self.buf.extend(b"\0" * (off - len(self.buf)))
        self.buf.extend(data)
        self.regions[name] = {"offset": off, "size": len(data), "type": kind, "shape": list(shape)}
        return off

    def add_bf16(self, name: str, t: torch.Tensor) -> int:
        return self.add(name, _bf16_bytes(t), "bf16", t.shape)

    def add_quant(self, name: str, w: torch.Tensor, precision: str, block_size: int):
        """Pack a (N, K) linear weight as <name>_scale + <name>_data."""
        N, K = w.shape
        assert N % 64 == 0 and K % 64 == 0, f"{name}: ({N}, {K}) must be 64-aligned"
        data, scale = quant_lib.quantize(precision, w.to(torch.bfloat16), block_size=block_size)
        self.add(name + "_scale", scale, "bf16_scale", (N, K // block_size))
        self.add(name + "_data", data, precision, (N, K))

    def finish(self):
        return bytes(self.buf), self.regions


# ----------------------------------------------------------------------------
# Text encoder (Qwen3-4B, 27 layers)
# ----------------------------------------------------------------------------

TE_LAYER_LINEARS = [
    ("q", "self_attn.q_proj.weight"),
    ("k", "self_attn.k_proj.weight"),
    ("v", "self_attn.v_proj.weight"),
    ("o", "self_attn.o_proj.weight"),
    ("gate", "mlp.gate_proj.weight"),
    ("up", "mlp.up_proj.weight"),
    ("down", "mlp.down_proj.weight"),
]
TE_LAYER_NORMS = [
    ("attn_norm", "input_layernorm.weight"),
    ("ffn_norm", "post_attention_layernorm.weight"),
    ("q_norm", "self_attn.q_norm.weight"),
    ("k_norm", "self_attn.k_norm.weight"),
]


def _layer_layout(builder_fn, num_layers: int):
    """Pack ``num_layers`` identical layers; returns (bytes, layer_regions, stride).

    ``builder_fn(bb, i)`` adds layer i's regions to a fresh _BinBuilder; every
    layer must produce the same region table (asserted), so the stride is the
    aligned size of one layer."""
    layers = []
    first = None
    for i in range(num_layers):
        bb = _BinBuilder()
        builder_fn(bb, i)
        data, regions = bb.finish()
        if first is None:
            first = regions
        else:
            assert regions == first, f"layer {i} region layout differs from layer 0"
        layers.append(data)
    stride = _align(max(len(d) for d in layers))
    out = bytearray()
    for d in layers:
        out.extend(d)
        out.extend(b"\0" * (stride - len(d)))
    return bytes(out), first, stride


def build_text_encoder_bin(snap: str, cfg: dict, out_bin: str, out_meta: str, embed_bin: str,
                           precision: str = "if4", block_size: int = 64) -> dict:
    te = cfg["text_encoder"]
    n_layers = te["num_layers_used"]
    t0 = time.time()
    state = ref.load_safetensors_dir(os.path.join(snap, "text_encoder"))
    print(f"[weights] text encoder: {len(state)} tensors loaded in {time.time() - t0:.0f}s")

    def layer(bb: _BinBuilder, i: int):
        p = f"model.layers.{i}."
        for name, key in TE_LAYER_NORMS:
            bb.add_bf16(name, state[p + key])
        for name, key in TE_LAYER_LINEARS:
            bb.add_quant(name, state[p + key], precision, block_size)

    t0 = time.time()
    layers_bytes, layer_regions, stride = _layer_layout(layer, n_layers)
    print(f"[weights] text encoder: {n_layers} layers quantized ({precision}) in {time.time() - t0:.0f}s, "
          f"stride {stride / 1e6:.1f} MB, total {len(layers_bytes) / 1e9:.2f} GB")
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(layers_bytes)
    emb = state["model.embed_tokens.weight"]
    with open(embed_bin, "wb") as f:
        f.write(_bf16_bytes(emb))
    meta = {
        "precision": precision, "block_size": block_size,
        "num_layers": n_layers, "layer_stride": stride, "layer_regions": layer_regions,
        "size": len(layers_bytes),
        "embed_tokens": {"path": os.path.basename(embed_bin), "shape": list(emb.shape), "type": "bf16"},
        "source": os.path.join(snap, "text_encoder"),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


# ----------------------------------------------------------------------------
# Transformer
# ----------------------------------------------------------------------------

def permute_qk_rows(w: torch.Tensor, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """Permute the output rows of a (H*head_dim, K) Q or K projection per head."""
    N, K = w.shape
    assert N % head_dim == 0
    return w.view(N // head_dim, head_dim, K)[:, perm, :].reshape(N, K)


def permute_norm_gamma(g: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return g[perm]


DIT_NONLAYER = [
    ("x_embedder", "x_embedder.weight"),
    ("context_embedder", "context_embedder.weight"),
    ("time_l1", "time_guidance_embed.timestep_embedder.linear_1.weight"),
    ("time_l2", "time_guidance_embed.timestep_embedder.linear_2.weight"),
    ("mod_img", "double_stream_modulation_img.linear.weight"),
    ("mod_txt", "double_stream_modulation_txt.linear.weight"),
    ("mod_single", "single_stream_modulation.linear.weight"),
    ("norm_out", "norm_out.linear.weight"),
    ("proj_out", "proj_out.weight"),
]


def dit_double_block_tensors(state: dict, i: int, dit: dict, perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Logical tensors of double block i, already RoPE-permuted and split."""
    p = f"transformer_blocks.{i}."
    hd, Dm = dit["head_dim"], dit["mlp_hidden"]
    t = {}
    for s, (pq, pk, pv, po, pff, nq, nk) in (
        ("img", ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff", "attn.norm_q", "attn.norm_k")),
        ("txt", ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out", "ff_context",
                 "attn.norm_added_q", "attn.norm_added_k")),
    ):
        t[f"{s}_q"] = permute_qk_rows(state[p + pq + ".weight"], hd, perm)
        t[f"{s}_k"] = permute_qk_rows(state[p + pk + ".weight"], hd, perm)
        t[f"{s}_v"] = state[p + pv + ".weight"]
        t[f"{s}_o"] = state[p + po + ".weight"]
        w_in = state[p + pff + ".linear_in.weight"]
        t[f"{s}_ff_gate"] = w_in[:Dm]
        t[f"{s}_ff_up"] = w_in[Dm:]
        t[f"{s}_ff_out"] = state[p + pff + ".linear_out.weight"]
        t[f"{s}_norm_q"] = permute_norm_gamma(state[p + nq + ".weight"], perm)
        t[f"{s}_norm_k"] = permute_norm_gamma(state[p + nk + ".weight"], perm)
    return t


def dit_single_block_tensors(state: dict, i: int, dit: dict, perm: torch.Tensor) -> dict[str, torch.Tensor]:
    p = f"single_transformer_blocks.{i}."
    D, hd, Dm = dit["hidden"], dit["head_dim"], dit["mlp_hidden"]
    w = state[p + "attn.to_qkv_mlp_proj.weight"]            # (3D + 2Dm, D)
    wo = state[p + "attn.to_out.weight"]                     # (D, D + Dm)
    return {
        "q": permute_qk_rows(w[0:D], hd, perm),
        "k": permute_qk_rows(w[D:2 * D], hd, perm),
        "v": w[2 * D:3 * D],
        "mlp_gate": w[3 * D:3 * D + Dm],
        "mlp_up": w[3 * D + Dm:3 * D + 2 * Dm],
        "out_attn": wo[:, :D].contiguous(),
        "out_mlp": wo[:, D:].contiguous(),
        "norm_q": permute_norm_gamma(state[p + "attn.norm_q.weight"], perm),
        "norm_k": permute_norm_gamma(state[p + "attn.norm_k.weight"], perm),
    }


def build_dit_bin(snap: str, cfg: dict, out_bin: str, out_meta: str,
                  precision: str = "if4", block_size: int = 64, bf16_keys=()) -> dict:
    dit = cfg["dit"]
    hd = dit["head_dim"]
    perm = ref.rotate_half_permutation(hd)
    t0 = time.time()
    state = ref.load_safetensors_dir(os.path.join(snap, "transformer"))
    print(f"[weights] transformer: {len(state)} tensors loaded in {time.time() - t0:.0f}s")
    bf16_keys = set(bf16_keys)

    nl = _BinBuilder()
    for name, key in DIT_NONLAYER:
        w = state[key]
        if key in bf16_keys:
            nl.add_bf16(name, w)
        else:
            nl.add_quant(name, w, precision, block_size)
    nl_bytes, nl_regions = nl.finish()

    def double(bb: _BinBuilder, i: int):
        t = dit_double_block_tensors(state, i, dit, perm)
        for k, v in t.items():
            if v.dim() == 1:
                bb.add_bf16(k, v)
            else:
                bb.add_quant(k, v, precision, block_size)

    def single(bb: _BinBuilder, i: int):
        t = dit_single_block_tensors(state, i, dit, perm)
        for k, v in t.items():
            if v.dim() == 1:
                bb.add_bf16(k, v)
            else:
                bb.add_quant(k, v, precision, block_size)

    t0 = time.time()
    dbl_bytes, dbl_regions, dbl_stride = _layer_layout(double, dit["num_double_blocks"])
    print(f"[weights] {dit['num_double_blocks']} double blocks quantized in {time.time() - t0:.0f}s "
          f"(stride {dbl_stride / 1e6:.1f} MB)")
    t0 = time.time()
    sgl_bytes, sgl_regions, sgl_stride = _layer_layout(single, dit["num_single_blocks"])
    print(f"[weights] {dit['num_single_blocks']} single blocks quantized in {time.time() - t0:.0f}s "
          f"(stride {sgl_stride / 1e6:.1f} MB)")

    buf = bytearray(nl_bytes)
    dbl_off = _align(len(buf)); buf.extend(b"\0" * (dbl_off - len(buf))); buf.extend(dbl_bytes)
    sgl_off = _align(len(buf)); buf.extend(b"\0" * (sgl_off - len(buf))); buf.extend(sgl_bytes)
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(buf)
    meta = {
        "precision": precision, "block_size": block_size, "bf16_keys": sorted(bf16_keys),
        "rope_perm": perm.tolist(),
        "nonlayer_regions": nl_regions,
        "double": {"base": dbl_off, "count": dit["num_double_blocks"], "stride": dbl_stride, "regions": dbl_regions},
        "single": {"base": sgl_off, "count": dit["num_single_blocks"], "stride": sgl_stride, "regions": sgl_regions},
        "size": len(buf),
        "source": os.path.join(snap, "transformer"),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[weights] transformer image {len(buf) / 1e9:.3f} GB -> {out_bin}")
    return meta


# ----------------------------------------------------------------------------
# VAE decoder (IF8 conv codes with per-(oc, tap, 64-ch block) scales)
# ----------------------------------------------------------------------------

def quantize_conv_if8(w: torch.Tensor):
    """(OC, C, kh, kw) bf16 -> int8 codes (same shape) and signed bf16 scales of
    shape (OC, kh*kw*ceil(C/64)) in [oc, ky, kx, ct] order (the
    ``run_conv2d_layer(block_scales=...)`` contract). Negative = INT8 codebook.
    Returns (codes, block_scales, max_rel_err)."""
    OC, C, kh, kw = w.shape
    ct = (C + 63) // 64
    Cp = ct * 64
    wp = torch.zeros(OC, Cp, kh, kw, dtype=torch.float32)
    wp[:, :C] = w.float()
    blocks = wp.view(OC, ct, 64, kh, kw).permute(0, 3, 4, 1, 2).reshape(OC, kh * kw * ct, 64)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-8)
    scale = (amax / 127.0).to(torch.bfloat16)
    q = (blocks / scale.float()[..., None]).round().clamp(-127, 127)
    recon = q * scale.float()[..., None]
    rel = ((blocks - recon).pow(2).sum() / blocks.pow(2).sum().clamp(min=1e-20)).sqrt()
    codes = q.view(OC, kh, kw, ct, 64).permute(0, 3, 4, 1, 2).reshape(OC, Cp, kh, kw)[:, :C]
    return codes.to(torch.int8).contiguous(), (-scale).contiguous(), float(rel)


def dequantize_conv_if8(codes: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    OC, C, kh, kw = codes.shape
    ct = (C + 63) // 64
    Cp = ct * 64
    cp = torch.zeros(OC, Cp, kh, kw, dtype=torch.float32)
    cp[:, :C] = codes.float()
    blocks = cp.view(OC, ct, 64, kh, kw).permute(0, 3, 4, 1, 2).reshape(OC, kh * kw * ct, 64)
    recon = blocks * block_scales.float().abs()[..., None]
    return recon.view(OC, kh, kw, ct, 64).permute(0, 3, 4, 1, 2).reshape(OC, Cp, kh, kw)[:, :C].contiguous()


def vae_conv_names(cfg: dict) -> list[tuple[str, str, int]]:
    """(weight key prefix, kind, pad) for every conv in decode order."""
    v = cfg["vae"]
    rev = list(reversed(v["block_out_channels"]))
    n_res = v["layers_per_block"] + 1
    names = [("post_quant_conv", "conv", 0), ("decoder.conv_in", "conv", 1)]
    def resnet(p, cin, cout):
        names.append((p + ".conv1", "conv", 1))
        names.append((p + ".conv2", "conv", 1))
        if cin != cout:
            names.append((p + ".conv_shortcut", "conv", 0))
    c = rev[0]
    resnet("decoder.mid_block.resnets.0", c, c)
    for k in ("to_q", "to_k", "to_v", "to_out.0"):
        names.append((f"decoder.mid_block.attentions.0.{k}", "linear", 0))
    resnet("decoder.mid_block.resnets.1", c, c)
    for i, cout in enumerate(rev):
        for j in range(n_res):
            resnet(f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
        c = cout
        if i < len(rev) - 1:
            names.append((f"decoder.up_blocks.{i}.upsamplers.0.conv", "conv", 1))
    names.append(("decoder.conv_out", "conv", 1))
    return names


def build_vae_weights(snap: str, cfg: dict, out_path: str) -> dict:
    state = ref.load_safetensors_dir(os.path.join(snap, "vae"))
    out = {"convs": {}, "norms": {}, "bn": {}, "precision": "if8+bf16"}      # both forms; the config picks
    worst = ("", 0.0)
    for key, kind, pad in vae_conv_names(cfg):
        w = state[key + ".weight"]
        if kind == "linear":
            w = w[:, :, None, None]
        codes, scales, rel = quantize_conv_if8(w)
        out["convs"][key] = {"codes": codes, "block_scales": scales,
                             "weight": w.to(torch.bfloat16).contiguous(),        # exact: the checkpoint is BF16
                             "bias": state[key + ".bias"].to(torch.bfloat16).contiguous(),
                             "pad": pad, "kind": kind, "rel_err": rel}
        if rel > worst[1]:
            worst = (key, rel)
    for k, v in state.items():
        if k.startswith("decoder.") and (k.endswith("norm1.weight") or k.endswith("norm2.weight")
                                          or k.endswith("group_norm.weight") or k.endswith("conv_norm_out.weight")):
            p = k[:-len(".weight")]
            out["norms"][p] = {"gamma": v.to(torch.bfloat16).contiguous(),
                               "beta": state[p + ".bias"].to(torch.bfloat16).contiguous()}
    out["bn"] = {"running_mean": state["bn.running_mean"].to(torch.bfloat16).contiguous(),
                 "running_var": state["bn.running_var"].to(torch.bfloat16).contiguous()}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"[weights] VAE decoder: {len(out['convs'])} convs (IF8 codes + BF16 weights), {len(out['norms'])} GroupNorms -> {out_path}; "
          f"worst conv rel err {worst[1]:.4f} ({worst[0]})")
    return out


# ----------------------------------------------------------------------------
# Shape-only layout planner (what the compiler addresses; must match the packer)
# ----------------------------------------------------------------------------

def _region_size(kind: str, shape, block_size: int = 64) -> int:
    n = 1
    for s in shape:
        n *= s
    if kind == "bf16":
        return n * 2
    if kind == "bf16_scale":
        return n * 2
    if kind in ("if4", "int4", "fp4"):
        return n // 2
    if kind in ("if8", "int8", "fp8"):
        return n
    raise ValueError(kind)


def plan_regions(spec, precision: str = "if4", block_size: int = 64):
    """``spec`` = [(name, kind, shape)] with kind in {"bf16", "quant"}; a "quant"
    entry expands to <name>_scale + <name>_data exactly like _BinBuilder.add_quant.
    Returns (regions, total_size) with the same offsets the packer produces."""
    regions = {}
    off = 0
    for name, kind, shape in spec:
        if kind == "quant":
            N, K = shape
            items = [(name + "_scale", "bf16_scale", (N, K // block_size)),
                     (name + "_data", precision, (N, K))]
        else:
            items = [(name, "bf16", tuple(shape))]
        for nm, kd, sh in items:
            off = _align(off)
            size = _region_size(kd, sh, block_size)
            regions[nm] = {"offset": off, "size": size, "type": kd, "shape": list(sh)}
            off += size
    return regions, off


def te_layer_spec(cfg: dict):
    te = cfg["text_encoder"]
    H, nh, nkv, hd, I = te["hidden"], te["heads"], te["kv_heads"], te["head_dim"], te["intermediate"]
    spec = [(name, "bf16", (H,)) for name, _ in TE_LAYER_NORMS[:2]]
    spec += [(name, "bf16", (hd,)) for name, _ in TE_LAYER_NORMS[2:]]
    shapes = {"q": (nh * hd, H), "k": (nkv * hd, H), "v": (nkv * hd, H), "o": (H, nh * hd),
              "gate": (I, H), "up": (I, H), "down": (H, I)}
    spec += [(name, "quant", shapes[name]) for name, _ in TE_LAYER_LINEARS]
    return spec


def dit_nonlayer_spec(cfg: dict):
    d = cfg["dit"]
    D, Dm, J, Cin, T = d["hidden"], d["mlp_hidden"], d["joint_dim"], d["in_channels"], d["time_channels"]
    shapes = {"x_embedder": (D, Cin), "context_embedder": (D, J), "time_l1": (D, T), "time_l2": (D, D),
              "mod_img": (6 * D, D), "mod_txt": (6 * D, D), "mod_single": (3 * D, D),
              "norm_out": (2 * D, D), "proj_out": (Cin, D)}
    bf16 = set(cfg["quantization"]["bf16_keys"])
    return [(name, "bf16" if key in bf16 else "quant", shapes[name]) for name, key in DIT_NONLAYER]


def dit_double_spec(cfg: dict):
    d = cfg["dit"]
    D, Dm, hd = d["hidden"], d["mlp_hidden"], d["head_dim"]
    spec = []
    for s in ("img", "txt"):
        spec += [(f"{s}_q", "quant", (D, D)), (f"{s}_k", "quant", (D, D)), (f"{s}_v", "quant", (D, D)),
                 (f"{s}_o", "quant", (D, D)), (f"{s}_ff_gate", "quant", (Dm, D)), (f"{s}_ff_up", "quant", (Dm, D)),
                 (f"{s}_ff_out", "quant", (D, Dm)), (f"{s}_norm_q", "bf16", (hd,)), (f"{s}_norm_k", "bf16", (hd,))]
    return spec


def dit_single_spec(cfg: dict):
    d = cfg["dit"]
    D, Dm, hd = d["hidden"], d["mlp_hidden"], d["head_dim"]
    return [("q", "quant", (D, D)), ("k", "quant", (D, D)), ("v", "quant", (D, D)),
            ("mlp_gate", "quant", (Dm, D)), ("mlp_up", "quant", (Dm, D)),
            ("out_attn", "quant", (D, D)), ("out_mlp", "quant", (D, Dm)),
            ("norm_q", "bf16", (hd,)), ("norm_k", "bf16", (hd,))]


def plan_te_layout(cfg: dict) -> dict:
    q = cfg["quantization"]
    regions, size = plan_regions(te_layer_spec(cfg), q["precision"], q["block_size"])
    stride = _align(size)
    n = cfg["text_encoder"]["num_layers_used"]
    return {"num_layers": n, "layer_stride": stride, "layer_regions": regions, "size": n * stride}


def plan_dit_layout(cfg: dict) -> dict:
    q = cfg["quantization"]
    d = cfg["dit"]
    nl, nl_size = plan_regions(dit_nonlayer_spec(cfg), q["precision"], q["block_size"])
    dbl, dbl_size = plan_regions(dit_double_spec(cfg), q["precision"], q["block_size"])
    sgl, sgl_size = plan_regions(dit_single_spec(cfg), q["precision"], q["block_size"])
    dbl_stride, sgl_stride = _align(dbl_size), _align(sgl_size)
    dbl_base = _align(nl_size)
    sgl_base = _align(dbl_base + d["num_double_blocks"] * dbl_stride)
    return {
        "nonlayer_regions": nl,
        "double": {"base": dbl_base, "count": d["num_double_blocks"], "stride": dbl_stride, "regions": dbl},
        "single": {"base": sgl_base, "count": d["num_single_blocks"], "stride": sgl_stride, "regions": sgl},
        "size": sgl_base + d["num_single_blocks"] * sgl_stride,
    }


def check_meta_matches_plan(meta: dict, plan: dict, what: str) -> None:
    """The packed bin's manifest must equal the shape-only plan the compiler used."""
    def norm(r):
        return {k: {"offset": v["offset"], "size": v["size"], "type": v["type"]} for k, v in r.items()}
    if "layer_regions" in plan:
        ok = (meta["layer_stride"] == plan["layer_stride"] and meta["num_layers"] == plan["num_layers"]
              and norm(meta["layer_regions"]) == norm(plan["layer_regions"]))
    else:
        ok = (norm(meta["nonlayer_regions"]) == norm(plan["nonlayer_regions"])
              and all(meta[k][f] == plan[k][f] for k in ("double", "single") for f in ("base", "count", "stride"))
              and all(norm(meta[k]["regions"]) == norm(plan[k]["regions"]) for k in ("double", "single")))
    if not ok:
        raise RuntimeError(f"{what}: packed bin layout differs from the planned layout; "
                           f"re-export the weights (flux2_klein_4b_weights.py --force)")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def load_config(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "flux2_klein_4b_config.json")) as f:
        return json.load(f)


def build_all(cfg: dict, snap: str | None = None, force: bool = False, which=("te", "dit", "vae")) -> None:
    snap = snap or ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
    q = cfg["quantization"]
    paths = {k: os.path.join(HERE, v) for k, v in cfg["paths"].items() if k.endswith(("_bin", "_meta", "vae_weights"))}
    if "te" in which and (force or not os.path.exists(paths["params_te_bin"])):
        build_text_encoder_bin(snap, cfg, paths["params_te_bin"], paths["params_te_meta"],
                               paths["embed_tokens_bin"], q["precision"], q["block_size"])
    if "dit" in which and (force or not os.path.exists(paths["params_dit_bin"])):
        build_dit_bin(snap, cfg, paths["params_dit_bin"], paths["params_dit_meta"],
                      q["precision"], q["block_size"], q["bf16_keys"])
    if "vae" in which and (force or not os.path.exists(paths["vae_weights"])):
        build_vae_weights(snap, cfg, paths["vae_weights"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export FLUX.2 klein 4B weights for the Unified Engine")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=["te", "dit", "vae"], choices=["te", "dit", "vae"])
    args = ap.parse_args()
    build_all(load_config(), force=args.force, which=tuple(args.only))
