"""
Pure PyTorch reimplementation of pi0.5 (LIBERO checkpoint) forward pass,
independent of JAX/openpi, for CPU/CUDA testing at different quant formats
(fp32/bf16/IF4) before FPGA bring-up.

Weights are loaded from models/pi05_libero/weights_export/*.npy (already
exported from the real gs://openpi-assets/checkpoints/pi05_libero checkpoint).

Usage:
    python pi05_torch_ref.py --device cuda --quant if4
    python pi05_torch_ref.py --device cpu  --quant fp32
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WEIGHTS_DIR = Path(__file__).parent / "weights_export"

# ---------------------------------------------------------------------------
# IF4 quantization: block_size=64, 4-bit signed code in [-8,7], one fp32 scale
# per 64-element block along the innermost (K) dimension. Matches the engine's
# q4_64 scheme (pi05_libero_config.json -> quantization.q4_64).
# ---------------------------------------------------------------------------
IF4_BLOCK = 64
IF4_MIN, IF4_MAX = -8, 7


def if4_fake_quantize(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantize the last dim of `w` in blocks of 64: quantize to 4-bit
    signed codes with a per-block scale, then immediately dequantize back to
    float. Returns a tensor of the same shape/dtype, with IF4 rounding error
    applied — this is what the model actually "sees" at IF4 precision."""
    orig_shape = w.shape
    orig_dtype = w.dtype
    K = orig_shape[-1]
    pad = (-K) % IF4_BLOCK
    if pad:
        w = F.pad(w, (0, pad))
    flat = w.reshape(-1, w.shape[-1] // IF4_BLOCK, IF4_BLOCK).float()

    block_max = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = block_max / IF4_MAX
    codes = torch.round(flat / scale).clamp(IF4_MIN, IF4_MAX)
    dequant = codes * scale

    dequant = dequant.reshape(w.shape)
    if pad:
        dequant = dequant[..., :K]
    return dequant.to(orig_dtype).reshape(orig_shape)


def _quantize_kernel_if4(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantize a JAX-style kernel (..., K_leading_dims, N) the way the
    hardware actually does: weight_init always reshapes to (K, N) then
    transposes to (N, K) before calling the real IF4 packer, so blocks of 64
    are grouped along the REDUCTION axis K. Calling if4_fake_quantize directly
    on the raw exported shape blocks along whatever axis happens to be last in
    that shape (almost always N, the output/head axis) -- a real, systematic
    mismatch versus hardware, not just rounding noise (verified: mean abs diff
    ~0.0037 on PaliGemma.img.embedding.kernel, values differing well outside
    bf16 precision). Reshaping to (K,N) then transposing before quantizing,
    like here, fixes it for every kernel/linear/einsum weight in this file --
    per-layer/per-head leading dims stay safe to merge into one K-block axis
    because the true K (the axis adjacent to N) is always a multiple of 64
    (the IF4 block size), so block boundaries never straddle a layer/head
    boundary; if4_fake_quantize's own internal zero-padding handles the one
    exception (patch-embed's K=588) exactly like weight_init's explicit pad."""
    orig_shape = w.shape
    N = orig_shape[-1]
    w2d = w.reshape(-1, N)                      # (K, N)
    wT = w2d.transpose(0, 1).contiguous()       # (N, K) -- K now last, correct block axis
    wT_q = if4_fake_quantize(wT)
    w2d_q = wT_q.transpose(0, 1).contiguous()   # back to (K, N)
    return w2d_q.reshape(orig_shape)


def maybe_quantize(w: torch.Tensor, quant: str) -> torch.Tensor:
    if quant == "if4":
        return _quantize_kernel_if4(w)
    return w  # fp32 / bf16 handled by .to(dtype) at load time


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
class Weights:
    def __init__(self, device: str, dtype: torch.dtype, quant: str):
        self.device = device
        self.dtype = dtype
        self.quant = quant
        manifest = json.loads((WEIGHTS_DIR / "manifest.json").read_text())
        self._cache = {}
        self._manifest = manifest

    def __getitem__(self, name: str) -> torch.Tensor:
        if name in self._cache:
            return self._cache[name]
        info = self._manifest[name]
        arr = np.load(WEIGHTS_DIR / info["file"])
        t = torch.from_numpy(arr).to(self.dtype)
        # Only quantize the big matmul weight matrices (kernels/linear), not
        # norms/biases/embeddings (biases and norm scales stay fp32/bf16 —
        # consistent with the engine's quantized-matmul kernels, which
        # dequantize weights but keep bias/accumulate in higher precision).
        if self.quant == "if4" and (name.endswith(".kernel") or name.endswith(".linear")
                                     or name.endswith(".w") or name.endswith("gating_einsum")):
            t = maybe_quantize(t, self.quant)
        t = t.to(self.device)
        self._cache[name] = t
        return t


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)
    out = normed * (1.0 + scale.float())
    return out.to(dtype)


def ada_rms_norm(x: torch.Tensor, cond: torch.Tensor, dense_w: torch.Tensor, dense_b: torch.Tensor,
                  eps: float = 1e-6):
    """AdaRMSNorm: RMSNorm (no learned scale) then scale/shift/gate from `cond`
    via a Dense(cond_dim -> 3*hidden) layer, split 3-way."""
    dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    normed = x32 * torch.rsqrt(var + eps)

    modulation = cond.float() @ dense_w.float() + dense_b.float()  # (b, 3*hidden)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = normed * (1.0 + scale)[:, None, :] + shift[:, None, :]
    return out.to(dtype), gate.to(dtype)


def gated_mlp(x: torch.Tensor, gating_einsum: torch.Tensor, linear: torch.Tensor) -> torch.Tensor:
    """gating_einsum: (2, d, mlp_dim) -> [gate_w, up_w]. linear: (mlp_dim, d)."""
    gate_w, up_w = gating_einsum[0], gating_einsum[1]
    gate = F.gelu(x @ gate_w, approximate="none")
    up = x @ up_w
    return (gate * up) @ linear


def sincos_pos_embed(t: torch.Tensor, dim: int, min_period=4e-3, max_period=4.0) -> torch.Tensor:
    """t: (b,) scalar timesteps in [0,1]. Returns (b, dim)."""
    device, dtype = t.device, torch.float32
    frac = torch.linspace(0.0, 1.0, dim // 2, device=device, dtype=dtype)
    period = min_period * (max_period / min_period) ** frac
    sinusoid = torch.einsum("b,d->bd", t.float(), 1.0 / period * 2 * math.pi)
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)


# ---------------------------------------------------------------------------
# Vision encoder: SigLIP So400m/14
# ---------------------------------------------------------------------------
def vision_encode(images: torch.Tensor, W: Weights) -> torch.Tensor:
    """images: (n_img, 224, 224, 3) float in [0,1] or [-1,1] (SigLIP expects
    roughly [-1,1] normalized). Returns (n_img, 256, 2048) projected tokens."""
    n_img = images.shape[0]
    patch_size = 14
    n_side = 224 // patch_size  # 16
    n_patches = n_side * n_side  # 256

    # Patch embed: conv with stride=kernel=14 == unfold + linear.
    kernel = W["PaliGemma.img.embedding.kernel"]  # (14,14,3,1152)
    bias = W["PaliGemma.img.embedding.bias"]      # (1152,)
    kernel_flat = kernel.reshape(-1, kernel.shape[-1])  # (14*14*3, 1152)

    # images: (n, 224, 224, 3) -> patches (n, 16, 16, 14, 14, 3) -> (n, 256, 14*14*3)
    x = images.reshape(n_img, n_side, patch_size, n_side, patch_size, 3)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(n_img, n_patches, patch_size * patch_size * 3)
    x = x @ kernel_flat + bias  # (n, 256, 1152)

    pos_embed = W["PaliGemma.img.pos_embedding"]  # (1, 256, 1152)
    x = x + pos_embed

    depth = 27
    for layer in range(depth):
        ln0_s = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][layer]
        ln0_b = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][layer]
        h = F.layer_norm(x.float(), (x.shape[-1],), ln0_s.float(), ln0_b.float()).to(x.dtype)

        q_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel"][layer]
        q_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias"][layer]
        k_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel"][layer]
        k_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias"][layer]
        v_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel"][layer]
        v_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias"][layer]
        o_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel"][layer]
        o_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias"][layer]

        num_heads, head_dim = 16, 72
        q = torch.einsum("ntd,dhk->nthk", h, q_k) + q_b
        k = torch.einsum("ntd,dhk->nthk", h, k_k) + k_b
        v = torch.einsum("ntd,dhk->nthk", h, v_k) + v_b
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))  # (n, heads, seq, head_dim)

        attn = F.scaled_dot_product_attention(q.float(), k.float(), v.float())  # bidirectional, no mask
        attn = attn.permute(0, 2, 1, 3).to(x.dtype)  # (n, seq, heads, head_dim)
        attn_out = torch.einsum("nthk,hkd->ntd", attn, o_k) + o_b

        x = x + attn_out

        ln1_s = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][layer]
        ln1_b = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][layer]
        h2 = F.layer_norm(x.float(), (x.shape[-1],), ln1_s.float(), ln1_b.float()).to(x.dtype)

        d0_k = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][layer]
        d0_b = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][layer]
        d1_k = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][layer]
        d1_b = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][layer]
        mlp_out = F.gelu(h2 @ d0_k + d0_b, approximate="tanh") @ d1_k + d1_b

        x = x + mlp_out

    enc_norm_s = W["PaliGemma.img.Transformer.encoder_norm.scale"]
    enc_norm_b = W["PaliGemma.img.Transformer.encoder_norm.bias"]
    x = F.layer_norm(x.float(), (x.shape[-1],), enc_norm_s.float(), enc_norm_b.float()).to(x.dtype)

    head_k = W["PaliGemma.img.head.kernel"]  # (1152, 2048)
    head_b = W["PaliGemma.img.head.bias"]
    x = x @ head_k + head_b  # (n_img, 256, 2048)
    return x


def vision_encode_checkpoints(images: torch.Tensor, W: Weights):
    """Same math as vision_encode, but a generator yielding (name, tensor) at
    every checkpoint the hardware side (compile_encoder's _debug_op calls)
    also stops at, in the same order. Used by debug_stepwise.py to compare a
    single micro-op's hardware output against the matching CPU tensor without
    re-deriving the whole forward pass by hand. n_img is fixed to 1 here (the
    debug harness runs one image slot at a time, matching compile_encoder's
    per-slot compile)."""
    assert images.shape[0] == 1, "vision_encode_checkpoints: one image slot at a time"
    patch_size = 14
    n_side = 224 // patch_size
    n_patches = n_side * n_side

    kernel = W["PaliGemma.img.embedding.kernel"]
    bias = W["PaliGemma.img.embedding.bias"]
    kernel_flat = kernel.reshape(-1, kernel.shape[-1])

    x = images.reshape(1, n_side, patch_size, n_side, patch_size, 3)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(1, n_patches, patch_size * patch_size * 3)
    x = x @ kernel_flat + bias

    pos_embed = W["PaliGemma.img.pos_embedding"]
    x = x + pos_embed
    yield "patch_embed_pos", x[0]

    depth = 27
    for layer in range(depth):
        ln0_s = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.scale"][layer]
        ln0_b = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_0.bias"][layer]
        h = F.layer_norm(x.float(), (x.shape[-1],), ln0_s.float(), ln0_b.float()).to(x.dtype)
        yield f"layer{layer}_ln0", h[0]

        q_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.kernel"][layer]
        q_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.query.bias"][layer]
        k_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.kernel"][layer]
        k_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.key.bias"][layer]
        v_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.kernel"][layer]
        v_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.value.bias"][layer]
        o_k = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel"][layer]
        o_b = W["PaliGemma.img.Transformer.encoderblock.MultiHeadDotProductAttention_0.out.bias"][layer]

        q = torch.einsum("ntd,dhk->nthk", h, q_k) + q_b
        k = torch.einsum("ntd,dhk->nthk", h, k_k) + k_b
        v = torch.einsum("ntd,dhk->nthk", h, v_k) + v_b
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))

        attn = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        attn = attn.permute(0, 2, 1, 3).to(x.dtype)
        attn_out = torch.einsum("nthk,hkd->ntd", attn, o_k) + o_b
        yield f"layer{layer}_o_proj", attn_out[0]

        x = x + attn_out
        yield f"layer{layer}_residual1", x[0]

        ln1_s = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.scale"][layer]
        ln1_b = W["PaliGemma.img.Transformer.encoderblock.LayerNorm_1.bias"][layer]
        h2 = F.layer_norm(x.float(), (x.shape[-1],), ln1_s.float(), ln1_b.float()).to(x.dtype)
        yield f"layer{layer}_ln1", h2[0]

        d0_k = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.kernel"][layer]
        d0_b = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_0.bias"][layer]
        d1_k = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel"][layer]
        d1_b = W["PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.bias"][layer]
        mlp_out = F.gelu(h2 @ d0_k + d0_b, approximate="tanh") @ d1_k + d1_b
        yield f"layer{layer}_mlp_out", mlp_out[0]

        x = x + mlp_out
        yield f"layer{layer}_residual2", x[0]

    enc_norm_s = W["PaliGemma.img.Transformer.encoder_norm.scale"]
    enc_norm_b = W["PaliGemma.img.Transformer.encoder_norm.bias"]
    x = F.layer_norm(x.float(), (x.shape[-1],), enc_norm_s.float(), enc_norm_b.float()).to(x.dtype)
    yield "encoder_norm", x[0]

    head_k = W["PaliGemma.img.head.kernel"]
    head_b = W["PaliGemma.img.head.bias"]
    x = x @ head_k + head_b
    yield "head_out", x[0]


# ---------------------------------------------------------------------------
# Joint prefix+suffix mixture-of-experts attention block (shared geometry:
# head_dim=256, num_heads=8, num_kv_heads=1 for both streams)
# ---------------------------------------------------------------------------
def joint_attention(pre_tokens, suf_tokens, attn_bias, layer, W, kv_cache=None):
    """pre_tokens: (b, P, 2048) or None (prefix already cached).
    suf_tokens: (b, S, 1024) or None.
    attn_bias: (b_or_1, Sq, Skv) additive mask (0 / -inf).
    kv_cache: optional (k_pre, v_pre) from a prior prefix-only pass, each
    (b, P, num_kv_heads, head_dim).
    Returns (pre_out, suf_out, new_kv_cache).
    """
    num_heads, num_kv_heads, head_dim = 8, 1, 256

    q_parts, k_parts, v_parts = [], [], []

    if pre_tokens is not None:
        q_ein = W["PaliGemma.llm.layers.attn.q_einsum.w"][layer]     # (8, 2048, 256)
        kv_ein = W["PaliGemma.llm.layers.attn.kv_einsum.w"][layer]   # (2, 1, 2048, 256)
        q_pre = torch.einsum("btd,hdk->bthk", pre_tokens, q_ein)
        k_pre = torch.einsum("btd,khdc->btkc", pre_tokens, kv_ein[0:1].permute(0, 1, 2, 3))
        # kv_einsum.w shape is (2, kv_heads, d, head_dim): [0]=k proj, [1]=v proj
        k_pre = torch.einsum("btd,hdc->bthc", pre_tokens, kv_ein[0])
        v_pre = torch.einsum("btd,hdc->bthc", pre_tokens, kv_ein[1])
        q_parts.append(q_pre)
        k_parts.append(k_pre)
        v_parts.append(v_pre)
    elif kv_cache is not None:
        k_parts.append(kv_cache[0][layer])
        v_parts.append(kv_cache[1][layer])

    if suf_tokens is not None:
        q_ein1 = W["PaliGemma.llm.layers.attn.q_einsum_1.w"][layer]   # (8, 1024, 256)
        kv_ein1 = W["PaliGemma.llm.layers.attn.kv_einsum_1.w"][layer]  # (2, 1, 1024, 256)
        q_suf = torch.einsum("btd,hdk->bthk", suf_tokens, q_ein1)
        k_suf = torch.einsum("btd,hdc->bthc", suf_tokens, kv_ein1[0])
        v_suf = torch.einsum("btd,hdc->bthc", suf_tokens, kv_ein1[1])

    k_all = torch.cat(k_parts + ([k_suf] if suf_tokens is not None else []), dim=1)  # (b, T_kv, kv_heads, hd)
    v_all = torch.cat(v_parts + ([v_suf] if suf_tokens is not None else []), dim=1)

    def run_attn(q):
        # q: (b, Tq, heads, hd) ; k_all/v_all: (b, Tkv, kv_heads, hd) with kv_heads=1 (MQA)
        b, Tq, h, hd = q.shape
        Tkv = k_all.shape[1]
        qf = q.permute(0, 2, 1, 3).float()                       # (b, h, Tq, hd)
        kf = k_all.permute(0, 2, 1, 3).float().expand(-1, h, -1, -1)  # (b, h, Tkv, hd) MQA broadcast
        vf = v_all.permute(0, 2, 1, 3).float().expand(-1, h, -1, -1)
        bias = attn_bias.float()
        if bias.dim() == 2:
            bias = bias[None, None]
        elif bias.dim() == 3:
            bias = bias[:, None]
        out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=bias)
        return out.permute(0, 2, 1, 3).to(q.dtype)  # (b, Tq, h, hd)

    pre_attn_out = None
    suf_attn_out = None
    if pre_tokens is not None:
        pre_attn_out = run_attn(q_pre)
        o_ein = W["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][layer]  # (8, 256, 2048)
        pre_attn_out = torch.einsum("bthk,hkd->btd", pre_attn_out, o_ein)
    if suf_tokens is not None:
        suf_attn_out = run_attn(q_suf)
        o_ein1 = W["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][layer]  # (8, 256, 1024)
        suf_attn_out = torch.einsum("bthk,hkd->btd", suf_attn_out, o_ein1)

    new_cache = None
    if pre_tokens is not None:
        new_cache = (k_pre, v_pre)  # caller collects per-layer into a list

    return pre_attn_out, suf_attn_out, new_cache


def prefix_forward(prefix_tokens, attn_bias, W, num_layers=18):
    """Runs the prefix once through all 18 layers, returns per-layer KV cache."""
    x = prefix_tokens
    k_cache, v_cache = [], []
    for layer in range(num_layers):
        norm_s = W["PaliGemma.llm.layers.pre_attention_norm.scale"][layer]
        h = rms_norm(x, norm_s)
        attn_out, _, (k_l, v_l) = joint_attention(h, None, attn_bias, layer, W)
        k_cache.append(k_l)
        v_cache.append(v_l)
        x = x + attn_out

        norm2_s = W["PaliGemma.llm.layers.pre_ffw_norm.scale"][layer]
        h2 = rms_norm(x, norm2_s)
        mlp_out = gated_mlp(h2, W["PaliGemma.llm.layers.mlp.gating_einsum"][layer],
                             W["PaliGemma.llm.layers.mlp.linear"][layer])
        x = x + mlp_out
    return x, (k_cache, v_cache)


def suffix_step(suf_tokens, adarms_cond, attn_bias, kv_cache, W, num_layers=18):
    x = suf_tokens
    for layer in range(num_layers):
        dense_w = W["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel"][layer]
        dense_b = W["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.bias"][layer]
        h, gate = ada_rms_norm(x, adarms_cond, dense_w, dense_b)

        _, attn_out, _ = joint_attention(None, h, attn_bias, layer, W, kv_cache=kv_cache)
        x = x + gate[:, None, :] * attn_out

        dense_w2 = W["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.kernel"][layer]
        dense_b2 = W["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.bias"][layer]
        h2, gate2 = ada_rms_norm(x, adarms_cond, dense_w2, dense_b2)

        mlp_out = gated_mlp(h2, W["PaliGemma.llm.layers.mlp_1.gating_einsum"][layer],
                             W["PaliGemma.llm.layers.mlp_1.linear"][layer])
        x = x + gate2[:, None, :] * mlp_out

    # Final AdaRMSNorm on the suffix stream (gemma.py Module.__call__: final_norms[1],
    # applied post-layers, before the caller's output projection). Missing this was a
    # real bug -- action_out_proj must consume the normed output, not raw residual.
    final_dense_w = W["PaliGemma.llm.final_norm_1.Dense_0.kernel"]
    final_dense_b = W["PaliGemma.llm.final_norm_1.Dense_0.bias"]
    x, _ = ada_rms_norm(x, adarms_cond, final_dense_w, final_dense_b)
    return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def build_prefix_attn_bias(n_prefix: int, device, dtype):
    """Prefix tokens attend bidirectionally to each other (no padding here;
    caller should pad+mask separately if seq < aligned length)."""
    return torch.zeros(n_prefix, n_prefix, device=device, dtype=dtype)


def build_suffix_attn_bias(n_prefix: int, n_suffix: int, device, dtype):
    """Suffix tokens attend to: all prefix tokens (bidirectional) + all suffix
    tokens (bidirectional within the action block, per pi0.5's ar_mask which
    is all-False beyond the first action token)."""
    return torch.zeros(n_suffix, n_prefix + n_suffix, device=device, dtype=dtype)


def run_pi05(images: torch.Tensor, prompt_tokens: torch.Tensor, state: torch.Tensor,
             W: Weights, num_denoise_steps: int = 10, action_horizon: int = 10, action_dim: int = 7,
             noise: torch.Tensor = None, return_kv: bool = False):
    device, dtype = W.device, W.dtype
    b = state.shape[0]

    # --- Vision ---
    vision_tokens = vision_encode(images, W)  # (n_img, 256, 2048)
    n_img = vision_tokens.shape[0]
    vision_tokens = vision_tokens.reshape(1, n_img * 256, 2048).expand(b, -1, -1)

    # --- Text embedding ---
    embed_table = W["PaliGemma.llm.embedder.input_embedding"]  # (257152, 2048)
    text_tokens = embed_table[prompt_tokens]  # (b, T_text, 2048)

    prefix_tokens = torch.cat([vision_tokens, text_tokens], dim=1)  # (b, P, 2048)
    n_prefix = prefix_tokens.shape[1]

    prefix_bias = build_prefix_attn_bias(n_prefix, device, torch.float32)
    _, kv_cache = prefix_forward(prefix_tokens, prefix_bias, W)

    # --- Action-expert flow-matching denoising loop ---
    dt = -1.0 / num_denoise_steps
    if noise is None:
        noise = torch.randn(b, action_horizon, action_dim, device=device, dtype=torch.float32)
    x_t = noise
    t = torch.ones(b, device=device, dtype=torch.float32)

    action_in_k, action_in_b = W["action_in_proj.kernel"], W["action_in_proj.bias"]
    time_in_k, time_in_b = W["time_mlp_in.kernel"], W["time_mlp_in.bias"]
    time_out_k, time_out_b = W["time_mlp_out.kernel"], W["time_mlp_out.bias"]
    action_out_k, action_out_b = W["action_out_proj.kernel"], W["action_out_proj.bias"]

    n_suffix = action_horizon
    suffix_bias = build_suffix_attn_bias(n_prefix, n_suffix, device, torch.float32)

    for _ in range(num_denoise_steps):
        action_tokens = (x_t.to(dtype) @ action_in_k) + action_in_b  # (b, ah, 1024)

        time_emb = sincos_pos_embed(t, dim=1024).to(dtype)
        time_emb = F.silu((time_emb @ time_in_k) + time_in_b)
        time_emb = F.silu((time_emb @ time_out_k) + time_out_b)  # (b, 1024) adarms_cond

        suf_out = suffix_step(action_tokens, time_emb, suffix_bias, kv_cache, W)
        v_t = (suf_out.float() @ action_out_k.float()) + action_out_b.float()  # (b, ah, action_dim)

        x_t = x_t + dt * v_t
        t = t + dt

    if return_kv:
        return x_t, kv_cache  # kv_cache = (k_cache, v_cache), each a list of 18 (b,P,1,256)
    return x_t  # (b, action_horizon, action_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quant", default="if4", choices=["fp32", "bf16", "if4"])
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    compute_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "if4": torch.bfloat16}[args.quant]
    W = Weights(device=args.device, dtype=compute_dtype, quant=args.quant)

    sample_dir = Path.home() / "apex-compute-ML" / "simple-llm" / "src" / "models" / "pi0_5" / "sample_data"
    img = np.load(sample_dir / f"sample_{args.sample}_image.npy").astype(np.float32) / 127.5 - 1.0
    wrist = np.load(sample_dir / f"sample_{args.sample}_wrist_image.npy").astype(np.float32) / 127.5 - 1.0
    img_t = torch.from_numpy(img).to(args.device, compute_dtype)
    wrist_t = torch.from_numpy(wrist).to(args.device, compute_dtype)
    pad_t = torch.zeros_like(img_t)
    images = torch.stack([img_t, wrist_t, pad_t], dim=0)
    # resize 256x256 -> 224x224 (SigLIP native res)
    images = images.permute(0, 3, 1, 2).float()
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    images = images.permute(0, 2, 3, 1).to(compute_dtype)

    meta = json.loads((sample_dir / "meta.json").read_text())
    state = torch.tensor([meta["state_example"]], device=args.device, dtype=torch.float32)

    # Placeholder prompt tokens (real SentencePiece tokenization would replace this;
    # ids just need to be valid indices into the 257152-row embedding table).
    prompt_tokens = torch.randint(0, 257152, (1, 16), device=args.device)

    torch.manual_seed(0)
    noise = torch.randn(1, 10, 7, device=args.device, dtype=torch.float32)
    # action_dim in the model is padded to 32 (action_in_proj.kernel is (32,1024));
    # only the first 7 dims are real (action_out_proj slices to 32, LiberoOutputs
    # then truncates to 7). Pad noise to 32 here to match.
    noise32 = torch.zeros(1, 10, 32, device=args.device, dtype=torch.float32)
    noise32[..., :7] = noise

    actions = run_pi05(images, prompt_tokens, state, W, noise=noise32)
    actions = actions[..., :7]

    torch.set_printoptions(precision=4, sci_mode=False)
    print(f"device={args.device} quant={args.quant}")
    print("action_chunk shape:", actions.shape)
    print(actions)


if __name__ == "__main__":
    main()
