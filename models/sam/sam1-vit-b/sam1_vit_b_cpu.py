#!/usr/bin/env python3
"""
SAM 1 ViT-B — CPU/GPU PyTorch reference implementation.

Loads weights from sam1_vit_b_bin/model.safetensors (HuggingFace facebook/sam-vit-base)
and runs full inference so you can compare against the FPGA accelerator.

Usage:
    python sam1_vit_b_cpu.py
    python sam1_vit_b_cpu.py --image ../../test_samples/vette.jpg --point 512 512
    python sam1_vit_b_cpu.py --image photo.jpg --point 300 200
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# SAM1 ViT-B config
# ---------------------------------------------------------------------------
IMG_SIZE     = 1024
PATCH_SIZE   = 16
EMBED_DIM    = 768
NUM_HEADS    = 12
DEPTH        = 12
MLP_RATIO    = 4.0
WINDOW_SIZE  = 14
GLOBAL_IDXS  = {2, 5, 8, 11}   # blocks with global attention
NECK_DIM     = 256
PROMPT_DIM   = 256

PIXEL_MEAN = torch.tensor([123.675, 116.28,  103.53 ]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([ 58.395,  57.12,   57.375]).view(3, 1, 1)


# ===========================================================================
# 1. IMAGE ENCODER (ViT-B + Neck)
# ===========================================================================

def window_partition(x, window_size):
    """[B,H,W,C] -> [B*nW, ws, ws, C], pad if needed. Returns (windows, pad_info)."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp, H, W)


def window_unpartition(windows, window_size, pad_info):
    """[B*nW, ws, ws, C] -> [B, H, W, C]"""
    Hp, Wp, H, W = pad_info
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    return x[:, :H, :W, :]


def get_rel_pos(q_size, k_size, rel_pos):
    """Resize and return relative positional bias [q_size, k_size, head_dim]."""
    max_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_dist:
        rel_pos = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_dist, mode="linear",
        ).permute(0, 2, 1).reshape(-1, rel_pos.shape[1])
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[relative.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """Add 2D decomposed relative positional bias to [B, nh, HW, HW] attention."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)   # [q_h, k_h, hd]
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)   # [q_w, k_w, hd]

    B_nW, nh, _, hd = q.shape
    q_r = q.reshape(B_nW, nh, q_h, q_w, hd)
    rel_h = torch.einsum("bnhwe,hke->bnhwk", q_r, Rh)   # [B_nW,nh,qh,qw,kh]
    rel_w = torch.einsum("bnhwe,wke->bnhwk", q_r, Rw)   # [B_nW,nh,qh,qw,kw]
    return (
        attn.reshape(B_nW, nh, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).reshape(B_nW, nh, q_h * q_w, k_h * k_w)


class ViTAttention(nn.Module):
    """ViT self-attention with optional relative position bias."""
    def __init__(self, dim, num_heads, use_rel_pos, input_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            H, W = input_size
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * H - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * W - 1, self.head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, nh, HW, hd]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)
        return self.proj(x)


class ViTBlock(nn.Module):
    """SAM1 ViT block. Attribute names match HF checkpoint keys directly."""
    def __init__(self, dim, num_heads, global_attn, window_size):
        super().__init__()
        input_size = (64, 64) if global_attn else (window_size, window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = ViTAttention(dim, num_heads, use_rel_pos=True, input_size=input_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * MLP_RATIO)
        # Named to match HF checkpoint: mlp.lin1, mlp.lin2
        self.mlp   = nn.ModuleDict({
            "lin1": nn.Linear(dim, hidden),
            "lin2": nn.Linear(hidden, dim),
        })
        self.window_size = 0 if global_attn else window_size
        self.global_attn = global_attn

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        H, W = x.shape[1], x.shape[2]

        if self.window_size > 0:
            x, pad_info = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_info)

        x = shortcut + x
        x = x + self.mlp["lin2"](F.gelu(self.mlp["lin1"](self.norm2(x))))
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias   = nn.Parameter(torch.zeros(C))
        self.eps    = eps

    def forward(self, x):  # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / (s + self.eps).sqrt() + self.bias[:, None, None]


class ImageEncoder(nn.Module):
    """ViT-B + neck. Attribute/sub-module names match HF checkpoint keys."""
    def __init__(self):
        super().__init__()
        # HF key: vision_encoder.patch_embed.projection.*
        self.patch_embed = nn.ModuleDict({"projection": nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, PATCH_SIZE)})
        self.pos_embed   = nn.Parameter(torch.zeros(1, 64, 64, EMBED_DIM))

        self.layers = nn.ModuleList([
            ViTBlock(EMBED_DIM, NUM_HEADS, global_attn=(i in GLOBAL_IDXS), window_size=WINDOW_SIZE)
            for i in range(DEPTH)
        ])

        # HF keys: vision_encoder.neck.conv1/conv2/layer_norm1/layer_norm2
        self.neck = nn.ModuleDict({
            "conv1":       nn.Conv2d(EMBED_DIM, NECK_DIM, 1, bias=False),
            "layer_norm1": LayerNorm2d(NECK_DIM),
            "conv2":       nn.Conv2d(NECK_DIM, NECK_DIM, 3, padding=1, bias=False),
            "layer_norm2": LayerNorm2d(NECK_DIM),
        })

    def forward(self, x):
        """[B, 3, 1024, 1024] -> [B, 256, 64, 64]"""
        x = self.patch_embed["projection"](x).permute(0, 2, 3, 1)  # [B, 64, 64, 768]
        x = x + self.pos_embed
        for blk in self.layers:
            x = blk(x)
        x = x.permute(0, 3, 1, 2)  # [B, 768, 64, 64]
        x = self.neck["layer_norm1"](self.neck["conv1"](x))
        x = self.neck["layer_norm2"](self.neck["conv2"](x))
        return x  # [B, 256, 64, 64]


# ===========================================================================
# 2. POSITIONAL ENCODING (shared between prompt encoder and image PE)
# ===========================================================================

class PositionEmbeddingRandom(nn.Module):
    """2D random Fourier feature positional encoding.
    Weights loaded from shared_image_embedding.positional_embedding [2, 128].
    """
    def __init__(self):
        super().__init__()
        # HF checkpoint key: shared_image_embedding.positional_embedding [2, 128]
        self.positional_embedding = nn.Parameter(torch.randn(2, PROMPT_DIM // 2))

    def _encode(self, coords):
        """coords [..., 2] in [0,1] -> [..., PROMPT_DIM]"""
        coords = 2.0 * coords - 1.0
        proj   = 2 * math.pi * coords @ self.positional_embedding.to(coords.dtype)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

    def forward_grid(self, size):
        """Encode a HxW grid -> [1, PROMPT_DIM, H, W]"""
        H, W   = size
        dev    = self.positional_embedding.device
        grid_y = (torch.arange(H, device=dev).float() + 0.5) / H   # [H]
        grid_x = (torch.arange(W, device=dev).float() + 0.5) / W   # [W]
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")      # [H, W] each
        coords = torch.stack([gx, gy], dim=-1)                       # [H, W, 2]
        return self._encode(coords).permute(2, 0, 1).unsqueeze(0)   # [1, 256, H, W]

    def forward_points(self, points):
        """points [B, N, 2] in normalized [0,1] -> [B, N, PROMPT_DIM]"""
        return self._encode(points)


# ===========================================================================
# 3. PROMPT ENCODER
# ===========================================================================

class PromptEncoder(nn.Module):
    def __init__(self, pe: PositionEmbeddingRandom):
        super().__init__()
        self.pe = pe
        # HF checkpoint keys: prompt_encoder.point_embed.{0..3}.weight
        self.point_embed       = nn.ModuleList([nn.Embedding(1, PROMPT_DIM) for _ in range(4)])
        self.not_a_point_embed = nn.Embedding(1, PROMPT_DIM)
        self.no_mask_embed     = nn.Embedding(1, PROMPT_DIM)
        # mask_embed not used (no mask prompt in our use case)

    def _encode_points(self, points, labels, image_size=(IMG_SIZE, IMG_SIZE)):
        """
        points: [B, N, 2]  pixel coords (0..image_size)
        labels: [B, N]     1=fg, 0=bg, -1=padding
        Returns [B, N, PROMPT_DIM]
        """
        # Normalize: shift to center of pixel, then divide by image size
        coords = (points + 0.5) / torch.tensor(
            [image_size[1], image_size[0]], dtype=torch.float32, device=points.device
        )
        pe = self.pe.forward_points(coords)       # [B, N, 256]

        B, N = labels.shape
        out  = torch.zeros_like(pe)
        for b in range(B):
            for n in range(N):
                lbl = int(labels[b, n].item())
                emb = pe[b, n]
                if lbl == 1:
                    emb = emb + self.point_embed[1].weight[0]
                elif lbl == 0:
                    emb = emb + self.point_embed[0].weight[0]
                else:
                    emb = self.not_a_point_embed.weight[0]
                out[b, n] = emb
        return out

    def forward(self, points, labels):
        """Returns (sparse [B,N,256], dense [B,256,64,64])"""
        B  = points.shape[0]
        sparse = self._encode_points(points, labels)
        dense  = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            B, -1, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE
        )
        return sparse, dense


# ===========================================================================
# 4. MASK DECODER
# ===========================================================================

class DecAttn(nn.Module):
    """SAM1 mask decoder attention (supports downsampled internal dim)."""
    def __init__(self, embed_dim, num_heads, internal_dim):
        super().__init__()
        self.num_heads    = num_heads
        self.internal_dim = internal_dim
        self.scale        = (internal_dim // num_heads) ** -0.5
        self.q_proj  = nn.Linear(embed_dim, internal_dim)
        self.k_proj  = nn.Linear(embed_dim, internal_dim)
        self.v_proj  = nn.Linear(embed_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embed_dim)

    def _split(self, x):
        B, N, _ = x.shape
        return x.reshape(B, N, self.num_heads, self.internal_dim // self.num_heads).transpose(1, 2)

    def forward(self, q, k, v):
        q = self._split(self.q_proj(q))
        k = self._split(self.k_proj(k))
        v = self._split(self.v_proj(v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out  = (attn.softmax(dim=-1) @ v).transpose(1, 2)
        B, N, nh, hd = out.shape
        return self.out_proj(out.reshape(B, N, nh * hd))


class TwoWayBlock(nn.Module):
    """SAM1 two-way attention block. Attribute names match HF checkpoint keys."""
    def __init__(self, skip_first_pe=False):
        super().__init__()
        d = PROMPT_DIM
        # self_attn: full 256-dim internal (downsample_rate=1)
        self.self_attn               = DecAttn(d, 8, d)
        self.layer_norm1             = nn.LayerNorm(d)
        # token -> image cross-attn: 128-dim internal (downsample_rate=2)
        self.cross_attn_token_to_image = DecAttn(d, 8, d // 2)
        self.layer_norm2             = nn.LayerNorm(d)
        # FFN: lin1/lin2 directly (match HF checkpoint: mlp.lin1, mlp.lin2)
        self.mlp                     = nn.ModuleDict({
            "lin1": nn.Linear(d, d * 8),
            "lin2": nn.Linear(d * 8, d),
        })
        self.layer_norm3             = nn.LayerNorm(d)
        # image -> token cross-attn
        self.cross_attn_image_to_token = DecAttn(d, 8, d // 2)
        self.layer_norm4             = nn.LayerNorm(d)
        self.skip_first_pe           = skip_first_pe

    def forward(self, queries, keys, query_pe, key_pe):
        # Self-attention on tokens
        if self.skip_first_pe:
            q_in = queries
        else:
            q_in = queries + query_pe
        queries = self.layer_norm1(queries + self.self_attn(q_in, q_in, queries))

        # Tokens cross-attend to image
        q = queries + query_pe
        k = keys    + key_pe
        queries = self.layer_norm2(queries + self.cross_attn_token_to_image(q, k, keys))

        # FFN
        mlp_out = self.mlp["lin2"](F.relu(self.mlp["lin1"](queries)))
        queries = self.layer_norm3(queries + mlp_out)

        # Image cross-attends to tokens
        q = keys    + key_pe
        k = queries + query_pe
        keys = self.layer_norm4(keys + self.cross_attn_image_to_token(q, k, queries))

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayBlock(skip_first_pe=(i == 0)) for i in range(2)
        ])
        self.final_attn_token_to_image = DecAttn(PROMPT_DIM, 8, PROMPT_DIM // 2)
        self.layer_norm_final_attn     = nn.LayerNorm(PROMPT_DIM)

    def forward(self, image_emb, image_pe, tokens):
        """
        image_emb: [B, 256, 64, 64]
        image_pe:  [B, 256, 64, 64]
        tokens:    [B, nq, 256]
        Returns: queries [B, nq, 256], keys [B, HW, 256]
        """
        B, C, H, W = image_emb.shape
        keys     = image_emb.flatten(2).permute(0, 2, 1)  # [B, HW, 256]
        key_pe   = image_pe.flatten(2).permute(0, 2, 1)   # [B, HW, 256]
        queries  = tokens
        query_pe = tokens  # SAM1 uses token embeddings as their own PE

        for layer in self.layers:
            queries, keys = layer(queries, keys, query_pe, key_pe)

        # Final cross-attn: tokens -> image
        q = queries + query_pe
        k = keys    + key_pe
        queries = self.layer_norm_final_attn(
            queries + self.final_attn_token_to_image(q, k, keys)
        )
        return queries, keys.reshape(B, H, W, C).permute(0, 3, 1, 2)


class HyperMLP(nn.Module):
    """3-layer MLP: proj_in -> layers.0 -> proj_out. Matches HF checkpoint keys."""
    def __init__(self, in_d, hidden_d, out_d):
        super().__init__()
        self.proj_in  = nn.Linear(in_d, hidden_d)
        self.layers   = nn.ModuleList([nn.Linear(hidden_d, hidden_d)])
        self.proj_out = nn.Linear(hidden_d, out_d)

    def forward(self, x):
        x = F.relu(self.proj_in(x))
        for l in self.layers:
            x = F.relu(l(x))
        return self.proj_out(x)


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = PROMPT_DIM
        self.iou_token   = nn.Embedding(1, d)
        self.mask_tokens = nn.Embedding(4, d)
        self.transformer = TwoWayTransformer()

        # Output upscaling: [B,256,64,64] -> [B,32,256,256]
        # HF keys: mask_decoder.upscale_conv1, upscale_layer_norm, upscale_conv2
        self.upscale_conv1      = nn.ConvTranspose2d(d, 64, 2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(64)
        self.upscale_conv2      = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # Hypernetwork MLPs: predict per-mask dot-product weights
        self.output_hypernetworks_mlps = nn.ModuleList([
            HyperMLP(d, d, 32) for _ in range(4)
        ])

        # IoU prediction head: HyperMLP(256, 256, 4)
        self.iou_prediction_head = HyperMLP(d, d, 4)

    def forward(self, image_emb, image_pe, sparse, dense):
        B = image_emb.shape[0]
        tokens = torch.cat([
            self.iou_token.weight.unsqueeze(0).expand(B, -1, -1),    # [B,1,256]
            self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1),  # [B,4,256]
            sparse,                                                     # [B,N,256]
        ], dim=1)

        src = image_emb + dense                          # [B,256,64,64]
        hs, src = self.transformer(src, image_pe, tokens)

        iou_tok      = hs[:, 0]                           # [B,256]
        mask_toks    = hs[:, 1:5]                         # [B,4,256]

        # Upscale
        up = F.gelu(self.upscale_layer_norm(self.upscale_conv1(src)))  # [B,64,128,128]
        up = F.gelu(self.upscale_conv2(up))                             # [B,32,256,256]

        # Hypernetwork dot-product
        hyper = torch.stack([self.output_hypernetworks_mlps[i](mask_toks[:, i])
                              for i in range(4)], dim=1)  # [B,4,32]
        B_, C_, H_, W_ = up.shape
        masks = (hyper @ up.flatten(2)).reshape(B_, 4, H_, W_)         # [B,4,256,256]

        iou = self.iou_prediction_head(iou_tok)           # [B,4]
        return masks, iou


# ===========================================================================
# 5. FULL MODEL
# ===========================================================================

class SAM1(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder  = ImageEncoder()
        self.shared_pe      = PositionEmbeddingRandom()
        self.prompt_encoder = PromptEncoder(self.shared_pe)
        self.mask_decoder   = MaskDecoder()

    def get_image_pe(self, dev):
        return self.shared_pe.forward_grid((64, 64)).to(dev)

    def forward(self, image, point_coords, point_labels):
        img_emb   = self.image_encoder(image)
        image_pe  = self.get_image_pe(image.device).expand(image.shape[0], -1, -1, -1)
        sparse, dense = self.prompt_encoder(point_coords, point_labels)
        masks, iou    = self.mask_decoder(img_emb, image_pe, sparse, dense)
        return masks, iou


# ===========================================================================
# 6. WEIGHT LOADING  (HuggingFace facebook/sam-vit-base checkpoint)
# ===========================================================================

def load_weights(model: SAM1, safetensors_path: str):
    """Map HF checkpoint keys to our model and load."""
    sd = load_file(safetensors_path)

    # Build full remap: checkpoint_key -> model_key
    remap = {}

    # Image encoder
    remap["vision_encoder.patch_embed.projection.weight"] = \
        "image_encoder.patch_embed.projection.weight"
    remap["vision_encoder.patch_embed.projection.bias"] = \
        "image_encoder.patch_embed.projection.bias"
    remap["vision_encoder.pos_embed"] = "image_encoder.pos_embed"

    for i in range(DEPTH):
        cp = f"vision_encoder.layers.{i}"
        mp = f"image_encoder.layers.{i}"
        for sfx in ["weight", "bias"]:
            # checkpoint uses layer_norm1/layer_norm2; model uses norm1/norm2
            remap[f"{cp}.layer_norm1.{sfx}"]   = f"{mp}.norm1.{sfx}"
            remap[f"{cp}.layer_norm2.{sfx}"]   = f"{mp}.norm2.{sfx}"
            remap[f"{cp}.attn.qkv.{sfx}"]      = f"{mp}.attn.qkv.{sfx}"
            remap[f"{cp}.attn.proj.{sfx}"]     = f"{mp}.attn.proj.{sfx}"
            remap[f"{cp}.mlp.lin1.{sfx}"]      = f"{mp}.mlp.lin1.{sfx}"
            remap[f"{cp}.mlp.lin2.{sfx}"]      = f"{mp}.mlp.lin2.{sfx}"
        remap[f"{cp}.attn.rel_pos_h"] = f"{mp}.attn.rel_pos_h"
        remap[f"{cp}.attn.rel_pos_w"] = f"{mp}.attn.rel_pos_w"

    for k in ["conv1.weight", "conv2.weight",
              "layer_norm1.weight", "layer_norm1.bias",
              "layer_norm2.weight", "layer_norm2.bias"]:
        remap[f"vision_encoder.neck.{k}"] = f"image_encoder.neck.{k}"

    # Shared PE
    remap["shared_image_embedding.positional_embedding"] = "shared_pe.positional_embedding"

    # Prompt encoder
    remap["prompt_encoder.no_mask_embed.weight"]     = "prompt_encoder.no_mask_embed.weight"
    remap["prompt_encoder.not_a_point_embed.weight"] = "prompt_encoder.not_a_point_embed.weight"
    for i in range(4):
        remap[f"prompt_encoder.point_embed.{i}.weight"] = \
            f"prompt_encoder.point_embed.{i}.weight"

    # Mask decoder
    remap["mask_decoder.iou_token.weight"]   = "mask_decoder.iou_token.weight"
    remap["mask_decoder.mask_tokens.weight"] = "mask_decoder.mask_tokens.weight"

    for k in ["upscale_conv1.weight", "upscale_conv1.bias",
              "upscale_conv2.weight", "upscale_conv2.bias",
              "upscale_layer_norm.weight", "upscale_layer_norm.bias"]:
        remap[f"mask_decoder.{k}"] = f"mask_decoder.{k}"

    for i in range(4):
        for k in ["proj_in.weight", "proj_in.bias",
                  "layers.0.weight", "layers.0.bias",
                  "proj_out.weight", "proj_out.bias"]:
            remap[f"mask_decoder.output_hypernetworks_mlps.{i}.{k}"] = \
                f"mask_decoder.output_hypernetworks_mlps.{i}.{k}"
    for k in ["proj_in.weight", "proj_in.bias",
              "layers.0.weight", "layers.0.bias",
              "proj_out.weight", "proj_out.bias"]:
        remap[f"mask_decoder.iou_prediction_head.{k}"] = \
            f"mask_decoder.iou_prediction_head.{k}"

    # Decoder transformer
    for li in range(2):
        cp = f"mask_decoder.transformer.layers.{li}"
        mp = f"mask_decoder.transformer.layers.{li}"
        for sub in ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]:
            for k in ["q_proj.weight", "q_proj.bias",
                      "k_proj.weight", "k_proj.bias",
                      "v_proj.weight", "v_proj.bias",
                      "out_proj.weight", "out_proj.bias"]:
                remap[f"{cp}.{sub}.{k}"] = f"{mp}.{sub}.{k}"
        for sfx in ["weight", "bias"]:
            for ln in ["layer_norm1", "layer_norm2", "layer_norm3", "layer_norm4"]:
                remap[f"{cp}.{ln}.{sfx}"] = f"{mp}.{ln}.{sfx}"
        for k in ["mlp.lin1.weight", "mlp.lin1.bias", "mlp.lin2.weight", "mlp.lin2.bias"]:
            remap[f"{cp}.{k}"] = f"{mp}.{k}"

    for k in ["q_proj.weight", "q_proj.bias",
              "k_proj.weight", "k_proj.bias",
              "v_proj.weight", "v_proj.bias",
              "out_proj.weight", "out_proj.bias"]:
        remap[f"mask_decoder.transformer.final_attn_token_to_image.{k}"] = \
            f"mask_decoder.transformer.final_attn_token_to_image.{k}"
    for sfx in ["weight", "bias"]:
        remap[f"mask_decoder.transformer.layer_norm_final_attn.{sfx}"] = \
            f"mask_decoder.transformer.layer_norm_final_attn.{sfx}"

    # Load
    model_sd = model.state_dict()
    loaded = 0
    warn   = 0
    for ck, mk in remap.items():
        if ck not in sd:
            print(f"  [WARN] ckpt missing: {ck}")
            warn += 1
            continue
        cv = sd[ck].float()
        if mk not in model_sd or cv.shape != model_sd[mk].shape:
            print(f"  [WARN] shape mismatch {ck}: ckpt={cv.shape} model={model_sd.get(mk,'?')}")
            warn += 1
            continue
        model_sd[mk] = cv
        loaded += 1

    model.load_state_dict(model_sd)
    print(f"  Loaded {loaded} tensors | warnings={warn}")


# ===========================================================================
# 7. PRE/POST PROCESSING
# ===========================================================================

def preprocess(img_path, device):
    img    = Image.open(img_path).convert("RGB")
    orig   = img.size  # (W, H)
    img    = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x      = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
    x      = (x - PIXEL_MEAN) / PIXEL_STD
    return x.unsqueeze(0).to(device), (orig[1], orig[0])  # tensor, (origH, origW)


def save_results(img_path, masks_logits, iou_scores, out_dir):
    """Save all 4 masks + best mask overlay."""
    os.makedirs(out_dir, exist_ok=True)
    best_idx = int(iou_scores.argmax())

    # Upsample all masks to 1024x1024
    masks_up = F.interpolate(
        masks_logits.unsqueeze(0).float(), size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False
    )[0]  # [4, 1024, 1024]

    binary = (masks_up > 0).cpu().numpy()  # [4, 1024, 1024]

    # Save each mask
    for i in range(4):
        path = os.path.join(out_dir, f"cpu_mask_{i}.png")
        Image.fromarray(binary[i].astype(np.uint8) * 255).save(path)

    # Best mask overlay
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).copy()
    mask = binary[best_idx]
    arr[mask, 1] = np.clip(arr[mask, 1].astype(int) + 100, 0, 255)
    Image.fromarray(arr).save(os.path.join(out_dir, "cpu_mask_overlay.png"))

    return binary[best_idx], os.path.join(out_dir, f"cpu_mask_{best_idx}.png")


# ===========================================================================
# 8. MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  default=None)
    parser.add_argument("--point",  nargs=2, type=int, default=[512, 512], metavar=("X", "Y"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"SAM1 ViT-B CPU reference  (device={device})")

    weights = os.path.join(SCRIPT_DIR, "sam1_vit_b_bin", "model.safetensors")
    if not os.path.exists(weights):
        print(f"  Weights not found: {weights}")
        sys.exit(1)

    # Find image
    if args.image:
        img_path = args.image
    else:
        for cand in [
            os.path.join(SCRIPT_DIR, "../../../test_samples/vette.jpg"),
            os.path.join(SCRIPT_DIR, "../../../test_samples/test_image.jpg"),
        ]:
            if os.path.exists(cand):
                img_path = cand
                break
        else:
            print("No image found. Pass --image <path>")
            sys.exit(1)

    out_dir = os.path.join(SCRIPT_DIR, "sam1_vit_b_bin")

    t_total = time.perf_counter()

    # Build & load
    t0 = time.perf_counter()
    model = SAM1().to(device).eval()
    print(f"  Model build:    {time.perf_counter()-t0:.3f}s")

    t0 = time.perf_counter()
    load_weights(model, weights)
    print(f"  Weight load:    {time.perf_counter()-t0:.3f}s")

    # Preprocess
    t0 = time.perf_counter()
    image_t, orig_hw = preprocess(img_path, device)
    print(f"  Preprocess:     {time.perf_counter()-t0:.3f}s  "
          f"img={os.path.basename(img_path)}  orig={orig_hw}")

    px, py = args.point
    point_coords = torch.tensor([[[px, py]]], dtype=torch.float32, device=device)
    point_labels = torch.tensor([[1]], dtype=torch.long, device=device)
    print(f"  Point prompt:   ({px}, {py})")

    # Inference
    with torch.no_grad():
        t0 = time.perf_counter()
        img_emb = model.image_encoder(image_t)
        t_enc   = time.perf_counter() - t0
        print(f"  Vision encoder: {t_enc:.3f}s  {tuple(img_emb.shape)}")

        t0      = time.perf_counter()
        image_pe = model.get_image_pe(device).expand(1, -1, -1, -1)
        sparse, dense = model.prompt_encoder(point_coords, point_labels)
        masks, iou = model.mask_decoder(img_emb, image_pe, sparse, dense)
        t_dec   = time.perf_counter() - t0
        print(f"  Mask decoder:   {t_dec:.3f}s  {tuple(masks.shape)}")

    iou_np  = iou[0].cpu().float().numpy()
    best    = int(iou_np.argmax())
    print(f"  IoU scores:     {iou_np.round(4)}")
    print(f"  Best mask:      idx={best}  IoU={iou_np[best]:.4f}")

    # Save
    t0 = time.perf_counter()
    binary, mask_path = save_results(img_path, masks[0], iou_np, out_dir)
    print(f"  Mask saved:     {mask_path}")
    print(f"  Postprocess:    {time.perf_counter()-t0:.3f}s")
    print(f"  Coverage:       {binary.mean()*100:.1f}%")
    print(f"  Total:          {time.perf_counter()-t_total:.3f}s")


if __name__ == "__main__":
    main()
