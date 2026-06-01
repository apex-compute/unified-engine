#!/usr/bin/env python3
"""LocateAnything-3B — own-PyTorch reference implementation (CPU/CUDA).

Mirrors the repo's *_cpu_test.py pattern (cf. mobilenetv2_cpu_test.py): a clean,
from-scratch forward pass in plain torch ops — no transformers model classes — so
it can later be ported to the UnifiedEngine / FPGA. Loads the stock HF weights and
parity-checks against captured reference activations (reference_capture.pt).

Architecture:
  MoonViT-SO-400M vision encoder  (27 layers, LayerNorm, GELU-tanh MLP, 2D RoPE,
                                   full bidirectional attn, learnable interp pos-emb)
    -> 2x2 patch merge (1152 -> 4608)
    -> MLP connector (LayerNorm(4608) -> Linear -> GELU -> Linear -> 2048)
    -> scatter into <IMG_CONTEXT> token positions
  Qwen2.5-3B LM (36 layers, GQA 16Q/2KV, 1D RoPE theta=1e6, SwiGLU, qkv bias)
    -> Parallel Box Decoding (slow/AR implemented; hybrid/MTP = TODO)

Usage:
  python locateanything_3b_cpu_test.py                 # parity + generate on vette.jpg
  python locateanything_3b_cpu_test.py --parity-only
  python locateanything_3b_cpu_test.py --image x.jpg --query car --device cuda
"""
import argparse
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "locateanything_3b_bin", "LocateAnything-3B")
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
IMG_CONTEXT = "<IMG_CONTEXT>"

PROMPTS = {
    "detect":       "Locate all the instances that matches the following description: {q}.",
    "ground_multi": "Locate all the instances that match the following description: {q}.",
    "ground_one":   "Locate a single instance that matches the following description: {q}.",
    "point":        "Point to: {q}.",
}


# ===========================================================================
# Image processor  (port of image_processing_locateanything.py)
# ===========================================================================

class ImageProcessor:
    def __init__(self, patch_size=14, in_token_limit=4096, merge_kernel_size=(2, 2),
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.patch_size = patch_size
        self.in_token_limit = in_token_limit
        self.merge_kernel_size = merge_kernel_size
        self.mean = mean
        self.std = std

    def _rescale(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        ps = self.patch_size
        if (w // ps) * (h // ps) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // ps) * (h // ps)))
            image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
        new_w, new_h = image.size
        pad_h = self.merge_kernel_size[0] * ps
        pad_w = self.merge_kernel_size[1] * ps
        target_w = math.ceil(new_w / pad_w) * pad_w
        target_h = math.ceil(new_h / pad_h) * pad_h
        if (target_w, target_h) != (new_w, new_h):
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        return image

    def __call__(self, image: Image.Image):
        from torchvision.transforms import functional as TF
        image = self._rescale(image)
        x = TF.to_tensor(image.convert("RGB"))
        x = TF.normalize(x, self.mean, self.std)
        ps = self.patch_size
        C, H, W = x.shape
        patches = x.reshape(C, H // ps, ps, W // ps, ps).permute(1, 3, 0, 2, 4)
        patches = patches.contiguous().view(-1, C, ps, ps)
        grid_hw = (H // ps, W // ps)
        return patches, grid_hw


# ===========================================================================
# MoonViT vision encoder
# ===========================================================================

class Learnable2DInterpPosEmb(nn.Module):
    def __init__(self, height, width, dim, mode="bicubic"):
        super().__init__()
        self.shape = (height, width)
        self.mode = mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))

    def forward(self, x, grid_hw):
        h, w = grid_hw
        if (h, w) == self.shape:
            pos = self.weight.flatten(end_dim=1)
        else:
            pos = (F.interpolate(self.weight.permute(2, 0, 1).unsqueeze(0),
                                 size=(h, w), mode=self.mode)
                   .squeeze(0).permute(1, 2, 0).flatten(end_dim=1))
        return x + pos


class PatchEmbed(nn.Module):
    def __init__(self, out_dim, patch_size, pos_h, pos_w):
        super().__init__()
        self.proj = nn.Conv2d(3, out_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Learnable2DInterpPosEmb(pos_h, pos_w, out_dim)

    def forward(self, x, grid_hw):
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_hw)


class Rope2D:
    """Precomputed separable 2D RoPE cis table (matches modeling_vit.Rope2DPosEmb)."""
    def __init__(self, dim, max_h=512, max_w=512, theta=10000):
        assert dim % 4 == 0
        self.dim, self.max_h, self.max_w, self.theta = dim, max_h, max_w, theta
        self._cis = None

    def _precompute(self, device):
        N = self.max_h * self.max_w
        flat = torch.arange(0, N).float().to(device)
        x_pos, y_pos = flat % self.max_w, flat // self.max_w
        dim_range = torch.arange(0, self.dim, 4)[: self.dim // 4].float().to(device)
        freqs = 1.0 / (self.theta ** (dim_range / self.dim))
        x_cis = torch.polar(torch.ones_like(torch.outer(x_pos, freqs)), torch.outer(x_pos, freqs))
        y_cis = torch.polar(torch.ones_like(torch.outer(y_pos, freqs)), torch.outer(y_pos, freqs))
        cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return cis.reshape(self.max_h, self.max_w, -1)

    def freqs_cis(self, grid_hw, device):
        if self._cis is None or self._cis.device != device:
            self._cis = self._precompute(device)
        h, w = grid_hw
        return self._cis[:h, :w].reshape(-1, self.dim // 2)


def apply_rope_2d(xq, xk, freqs_cis):
    # xq,xk: (L, heads, head_dim);  freqs_cis: (L, head_dim/2) complex64
    fc = freqs_cis.unsqueeze(-2)  # (L,1,hd/2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_ * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MLP2(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden)
        self.fc1 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc1(F.gelu(self.fc0(x), approximate="tanh"))


class MoonVitBlock(nn.Module):
    def __init__(self, hidden, num_heads, mlp_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.norm0 = nn.LayerNorm(hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.mlp = MLP2(hidden, mlp_dim, hidden)
        self.wqkv = nn.Linear(hidden, hidden * 3, bias=True)
        self.wo = nn.Linear(hidden, hidden, bias=True)

    def _attn(self, x, freqs_cis):
        L = x.shape[0]
        qkv = self.wqkv(x).view(L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)               # (L, heads, hd)
        q, k = apply_rope_2d(q, k, freqs_cis)
        # single image -> full bidirectional attention (one packed sequence)
        q = q.transpose(0, 1)                      # (heads, L, hd)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        out = F.scaled_dot_product_attention(q, k, v)   # no mask = full attn
        out = out.transpose(0, 1).reshape(L, -1)
        return self.wo(out)

    def forward(self, x, freqs_cis):
        x = x + self._attn(self.norm0(x), freqs_cis)
        x = x + self.mlp(self.norm1(x))
        return x


class MoonVitEncoder(nn.Module):
    def __init__(self, hidden, num_layers, num_heads, mlp_dim):
        super().__init__()
        self.rope_2d = Rope2D(hidden // num_heads)
        self.blocks = nn.ModuleList([MoonVitBlock(hidden, num_heads, mlp_dim)
                                     for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden)

    def forward(self, x, grid_hw):
        freqs_cis = self.rope_2d.freqs_cis(grid_hw, x.device)
        for blk in self.blocks:
            x = blk(x, freqs_cis)
        return self.final_layernorm(x)


def patch_merge(x, grid_hw, merge_kernel_size=(2, 2)):
    h, w = grid_hw
    kh, kw = merge_kernel_size
    d = x.size(-1)
    nh, nw = h // kh, w // kw
    x = x.view(nh, kh, nw, kw, d).permute(0, 2, 1, 3, 4).contiguous()
    return x.view(nh * nw, kh * kw * d)


class MoonVit(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v = cfg["vision_config"]
        self.merge_kernel_size = tuple(v["merge_kernel_size"])
        self.patch_embed = PatchEmbed(v["hidden_size"], v["patch_size"],
                                      v["init_pos_emb_height"], v["init_pos_emb_width"])
        self.encoder = MoonVitEncoder(v["hidden_size"], v["num_hidden_layers"],
                                      v["num_attention_heads"], v["intermediate_size"])

    def forward(self, pixel_values, grid_hw):
        x = self.patch_embed(pixel_values, grid_hw)
        x = self.encoder(x, grid_hw)
        return patch_merge(x, grid_hw, self.merge_kernel_size)


# ===========================================================================
# Qwen2.5-3B language model
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        t = cfg["text_config"]
        self.nh = t["num_attention_heads"]
        self.nkv = t["num_key_value_heads"]
        self.hd = t["head_dim"]
        self.q_proj = nn.Linear(t["hidden_size"], self.nh * self.hd, bias=True)
        self.k_proj = nn.Linear(t["hidden_size"], self.nkv * self.hd, bias=True)
        self.v_proj = nn.Linear(t["hidden_size"], self.nkv * self.hd, bias=True)
        self.o_proj = nn.Linear(self.nh * self.hd, t["hidden_size"], bias=False)

    def forward(self, x, cos, sin, kv_cache, causal):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        if kv_cache is not None:
            kv_cache["k"], kv_cache["v"] = k, v
        g = self.nh // self.nkv
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal and T > 1)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class Qwen2MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        t = cfg["text_config"]
        self.gate_proj = nn.Linear(t["hidden_size"], t["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(t["hidden_size"], t["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(t["intermediate_size"], t["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eps = cfg["text_config"]["rms_norm_eps"]
        h = cfg["text_config"]["hidden_size"]
        self.input_layernorm = RMSNorm(h, eps)
        self.self_attn = Qwen2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.mlp = Qwen2MLP(cfg)

    def forward(self, x, cos, sin, kv_cache, causal):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, causal)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        t = cfg["text_config"]
        self.embed_tokens = nn.Embedding(cfg["model"]["vocab_size"], t["hidden_size"])
        self.layers = nn.ModuleList([Qwen2Layer(cfg) for _ in range(t["num_hidden_layers"])])
        self.norm = RMSNorm(t["hidden_size"], t["rms_norm_eps"])
        self.hd = t["head_dim"]
        self.theta = t["rope_theta"]

    def _rope(self, position_ids, device, dtype):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))
        freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]   # (B,T,hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype)[:, None], emb.sin().to(dtype)[:, None]    # (B,1,T,hd)

    def forward(self, inputs_embeds, position_ids, kv_caches, causal):
        cos, sin = self._rope(position_ids, inputs_embeds.device, inputs_embeds.dtype)
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_caches[i] if kv_caches is not None else None, causal)
        return self.norm(x)


class LocateAnything(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision_model = MoonVit(cfg)
        c = cfg["connector"]
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(c["in_features"]),
            nn.Linear(c["in_features"], c["hidden_features"]),
            nn.GELU(),
            nn.Linear(c["hidden_features"], c["out_features"]),
        )
        self.language_model = _LM(cfg)
        self.image_token_index = cfg["model"]["image_token_index"]

    def vision_features(self, pixel_values, grid_hw):
        return self.vision_model(pixel_values, grid_hw)   # (n_merged, 4608)


class _LM(nn.Module):
    """Wrapper so weights load as language_model.model.* / language_model.lm_head.*"""
    def __init__(self, cfg):
        super().__init__()
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg["text_config"]["hidden_size"],
                                 cfg["model"]["vocab_size"], bias=False)


# ===========================================================================
# Box decode (port of generate_utils.py, greedy/argmax path)
# ===========================================================================

def build_token_ids(cfg):
    s = cfg["special_tokens"]
    return dict(s)


def draw_overlay(image_path, answer, out_path):
    """Draw the predicted boxes/points on the image and save it.

    Coordinates in the model output are normalized integers in [0, 1000] in
    xyxy order: <box><x1><y1><x2><y2></box> (4-tuple = box, 2-tuple = point).
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, H // 40))
    except Exception:
        font = ImageFont.load_default()

    label = (re.findall(r"<ref>(.*?)</ref>", answer) or [""])[0]
    color = (255, 40, 40)

    boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = (int(x1) / 1000 * W, int(y1) / 1000 * H,
                          int(x2) / 1000 * W, int(y2) / 1000 * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, H // 250))
        tag = f"{label} {i+1}" if label else f"{i+1}"
        draw.text((x1 + 3, max(0, y1 - (H // 35))), tag, fill=color, font=font)

    points = re.findall(r"<box><(\d+)><(\d+)></box>", answer)
    for (x, y) in points:
        px, py = int(x) / 1000 * W, int(y) / 1000 * H
        r = max(4, H // 120)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=3)

    img.save(out_path)
    return out_path, len(boxes), len(points)


_PALETTE = [(255, 40, 40), (40, 160, 255), (40, 200, 90), (255, 170, 0),
            (200, 60, 255), (0, 210, 210), (255, 90, 160), (150, 110, 60)]


def draw_overlay_multi(image_path, items, out_path):
    """Draw boxes/points from several queries on one image, each query a distinct
    color. `items` is a list of (label, answer) pairs. Returns (path, n_boxes)."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, H // 40))
    except Exception:
        font = ImageFont.load_default()

    total = 0
    for qi, (label, answer) in enumerate(items):
        color = _PALETTE[qi % len(_PALETTE)]
        ref = (re.findall(r"<ref>(.*?)</ref>", answer) or [label])[0]
        boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = (int(x1) / 1000 * W, int(y1) / 1000 * H,
                              int(x2) / 1000 * W, int(y2) / 1000 * H)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, H // 250))
            tag = f"{ref} {i+1}" if ref else f"{i+1}"
            draw.text((x1 + 3, max(0, y1 - (H // 35))), tag, fill=color, font=font)
            total += 1
        for (x, y) in re.findall(r"<box><(\d+)><(\d+)></box>", answer):
            px, py = int(x) / 1000 * W, int(y) / 1000 * H
            r = max(4, H // 120)
            draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=3)
    img.save(out_path)
    return out_path, total


def build_prompt_text(question, num_image_tokens):
    img_block = f"<image 1><img>{IMG_CONTEXT * num_image_tokens}</img>"
    return ("<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
            "<|im_start|>user\n" f"{img_block}{question}" "<|im_end|>\n"
            "<|im_start|>assistant\n")


# ===========================================================================
# Weight loading
# ===========================================================================

def ensure_hf_model(model_dir: str = MODEL_DIR, repo: str = "nvidia/LocateAnything-3B"):
    """Download the HF model into model_dir on first use (mirrors the repo's
    *_test.py download pattern). Verifies ALL weight shards listed in the
    safetensors index actually exist -- a config.json-only check would treat an
    interrupted/partial download as complete and never fetch the missing shard.
    snapshot_download resumes, so re-calling is safe."""
    import json as _json
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    complete = os.path.exists(os.path.join(model_dir, "config.json")) and os.path.exists(idx)
    if complete:
        shards = set(_json.load(open(idx))["weight_map"].values())
        missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
        if missing:
            print(f"  [download] incomplete: missing shards {missing} -> resuming")
            complete = False
    if complete:
        return model_dir
    from huggingface_hub import snapshot_download
    print(f"Downloading {repo} -> {os.path.abspath(model_dir)} ...")
    snapshot_download(repo_id=repo, local_dir=model_dir,
                      allow_patterns=["*.json", "*.txt", "*.py", "*.safetensors", "*.model"])
    print("Download complete.")
    return model_dir


def load_weights(model: LocateAnything, model_dir: str, device, dtype):
    ensure_hf_model(model_dir)
    from safetensors.torch import load_file
    import glob
    sd = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # tied embeddings: lm_head may equal embed_tokens; both are saved here though.
    real_missing = [m for m in missing if "rope" not in m]
    if real_missing:
        print(f"  [warn] {len(real_missing)} missing keys, e.g. {real_missing[:4]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:4]}")
    model.to(device=device, dtype=dtype)
    return model


# ===========================================================================
# Parity + generate
# ===========================================================================

@torch.no_grad()
def run_parity(model, cap, device, dtype):
    pv = cap["pixel_values"].to(device, dtype=dtype)
    gh, gw = int(cap["image_grid_hws"][0][0]), int(cap["image_grid_hws"][0][1])
    vit = model.vision_features(pv, (gh, gw))
    conn = model.mlp1(vit)

    def cmp(name, ours, ref):
        ours = ours.float().cpu()
        ref = ref.float().cpu()
        d = (ours - ref).abs()
        denom = ref.abs().mean().clamp_min(1e-9)
        print(f"  {name:14s} shape={tuple(ours.shape)} "
              f"mean|diff|={d.mean():.3e} max|diff|={d.max():.3e} rel={d.mean()/denom:.3e}")

    print("Stage parity vs stock HF reference:")
    cmp("vit_embeds", vit, cap["vit_embeds"])
    cmp("connector", conn, cap["connector_out"])


@torch.no_grad()
def generate_slow(model, tok, cfg, pixel_values, grid_hw, input_ids, device, dtype,
                  max_new_tokens=256, conn=None):
    tids = build_token_ids(cfg)
    im_end = tids["im_end_token_id"]

    if conn is None:                                        # vision depends only on the image
        vit = model.vision_features(pixel_values.to(device, dtype=dtype), grid_hw)
        conn = model.mlp1(vit)                              # (n_img, 2048)

    ids = input_ids.to(device)
    emb = model.language_model.model.embed_tokens(ids)       # (1, T, 2048)
    sel = (ids[0] == model.image_token_index)
    emb[0, sel] = conn[: int(sel.sum())].to(emb.dtype)

    n_layers = len(model.language_model.model.layers)
    kv = [{"k": None, "v": None} for _ in range(n_layers)]
    T = ids.shape[1]
    pos = torch.arange(T, device=device).unsqueeze(0)
    h = model.language_model.model(emb, pos, kv, causal=True)
    logits = model.language_model.lm_head(h[:, -1:])
    next_id = int(logits[0, -1].argmax())

    out = [next_id]
    cur = T
    while next_id != im_end and len(out) < max_new_tokens:
        tok_emb = model.language_model.model.embed_tokens(
            torch.tensor([[next_id]], device=device))
        pos = torch.tensor([[cur]], device=device)
        h = model.language_model.model(tok_emb, pos, kv, causal=False)
        logits = model.language_model.lm_head(h[:, -1:])
        next_id = int(logits[0, -1].argmax())
        out.append(next_id)
        cur += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(REPO_ROOT, "test_samples", "vette.jpg"))
    ap.add_argument("--prompt-kind", default="detect", choices=list(PROMPTS))
    ap.add_argument("--query", nargs="+", default=["car"],
                    help="one or more things to detect, e.g. --query car person 'traffic light'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--overlay", default=None,
                    help="output path for the box overlay image (default: <image>_boxes.jpg next to script)")
    ap.add_argument("--no-overlay", action="store_true", help="skip drawing the overlay")
    ap.add_argument("--capture", default=os.path.join(SCRIPT_DIR, "reference_capture.pt"))
    args = ap.parse_args()

    import json
    cfg = json.load(open(os.path.join(SCRIPT_DIR, "locateanything_3b_config.json")))
    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"building model on {device} ({args.dtype}) ...")
    model = LocateAnything(cfg).eval()
    load_weights(model, MODEL_DIR, device, dtype)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params: {nparams/1e9:.2f}B")

    if os.path.exists(args.capture):
        cap = torch.load(args.capture, map_location="cpu", weights_only=False)
        run_parity(model, cap, device, dtype)
    else:
        print(f"[skip parity] no capture at {args.capture}")

    if args.parity_only:
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    improc = ImageProcessor(merge_kernel_size=tuple(cfg["vision_config"]["merge_kernel_size"]))
    img = Image.open(args.image).convert("RGB")
    pixel_values, grid_hw = improc(img)
    mk = cfg["vision_config"]["merge_kernel_size"]
    n_img = (grid_hw[0] * grid_hw[1]) // (mk[0] * mk[1])
    print(f"grid={grid_hw} img_tokens={n_img}")

    # Vision features depend only on the image -> compute once, reuse per query.
    vit = model.vision_features(pixel_values.to(device, dtype=dtype), grid_hw)
    conn = model.mlp1(vit)

    items = []  # (query, answer)
    for q in args.query:
        text = build_prompt_text(PROMPTS[args.prompt_kind].format(q=q), n_img)
        input_ids = tok([text], return_tensors="pt").input_ids
        out_ids = generate_slow(model, tok, cfg, pixel_values, grid_hw, input_ids,
                                device, dtype, args.max_new_tokens, conn=conn)
        answer = tok.decode(out_ids, skip_special_tokens=False)
        boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
        print(f"\n=== query: {q!r} ===")
        print(answer)
        print(f"{len(boxes)} boxes: {boxes}")
        items.append((q, answer))

    if not args.no_overlay:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = args.overlay or os.path.join(SCRIPT_DIR, f"{stem}_boxes.jpg")
        path, nb = draw_overlay_multi(args.image, items, out_path)
        print(f"\noverlay saved -> {path}  ({nb} boxes across {len(items)} quer"
              f"{'y' if len(items)==1 else 'ies'})")


if __name__ == "__main__":
    main()
