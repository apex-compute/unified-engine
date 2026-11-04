#!/usr/bin/env python3
"""FLUX.2 [klein] 4B host reference (the numeric oracle for the FPGA port).

A dependency-light re-implementation of the diffusers ``Flux2KleinPipeline``
math in plain torch, driven straight from the safetensors checkpoint:

  * ``Qwen3TextEncoder``   -- Qwen3-4B (27 of 36 layers), hidden-state taps at
                              layers 9/18/27 concatenated to 7680 (transformers).
  * ``Flux2Transformer``   -- 5 double-stream + 20 single-stream blocks,
                              4-axis RoPE (theta 2000), shared adaLN modulation.
  * ``Flux2VaeDecoder``    -- AutoencoderKLFlux2 decoder (32-ch latent, 2x2
                              patchify, BatchNorm latent de-normalisation).
  * ``flow_match_sigmas``  -- FlowMatchEuler schedule with the empirical mu shift.

Everything is written as explicit tensor ops (no nn.Module graph) so the FPGA
compiler (``flux2_klein_4b_test.py``) can mirror it op-for-op, and so IF4 /
IF8 weight quantization can be *simulated* here by swapping tensors in the
state dicts (``fake_quantize_state``). ``test_flux2_klein_4b.py`` checks this
file against the official diffusers pipeline when diffusers is installed.

Layout conventions follow diffusers exactly (joint sequence = [text; image],
latents packed as (tokens, 128) with token = (h, w) row-major, channel =
c*4 + py*2 + px).
"""
from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

HF_REPO = "black-forest-labs/FLUX.2-klein-4B"

# ----------------------------------------------------------------------------
# Checkpoint access
# ----------------------------------------------------------------------------

def snapshot_dir(repo: str = HF_REPO, local_dir: str | None = None) -> str:
    """Return the local snapshot directory of the HF repo (downloading the
    diffusers folders if needed; the 7.8 GB single-file checkpoint is skipped)."""
    if local_dir and os.path.exists(os.path.join(local_dir, "model_index.json")):
        return local_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo, local_dir=local_dir,
        allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*",
                        "tokenizer/*", "transformer/*", "vae/*", "LICENSE.md"])


def load_safetensors_dir(folder: str, prefix: str | None = None,
                         device: str = "cpu") -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    out: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        for k, v in load_file(f, device=device).items():
            if prefix is None or k.startswith(prefix):
                out[k] = v
    if not out:
        raise FileNotFoundError(f"no safetensors found under {folder}")
    return out


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Geometry helpers (shared with the FPGA compiler)
# ----------------------------------------------------------------------------

@dataclass
class Flux2Geometry:
    """Everything that depends on the output resolution and text length."""
    height: int = 512
    width: int = 512
    max_sequence_length: int = 512
    num_inference_steps: int = 4
    vae_scale_factor: int = 8
    in_channels: int = 128          # DiT token width (= 32 latent ch * 2 * 2)
    hidden: int = 3072
    heads: int = 24
    head_dim: int = 128
    joint_dim: int = 7680           # 3 Qwen3 layers * 2560
    mlp_hidden: int = 9216          # mlp_ratio 3.0
    text_hidden: int = 2560
    text_layers_used: int = 27
    text_tap_layers: tuple = (9, 18, 27)
    rope_theta: float = 2000.0
    axes_dims_rope: tuple = (32, 32, 32, 32)
    time_channels: int = 256
    eps: float = 1e-6

    @property
    def latent_h(self) -> int:      # VAE latent (32-ch) grid
        return 2 * (self.height // (self.vae_scale_factor * 2))

    @property
    def latent_w(self) -> int:
        return 2 * (self.width // (self.vae_scale_factor * 2))

    @property
    def token_h(self) -> int:       # DiT image-token grid (after 2x2 patchify)
        return self.latent_h // 2

    @property
    def token_w(self) -> int:
        return self.latent_w // 2

    @property
    def n_img(self) -> int:
        return self.token_h * self.token_w

    @property
    def n_txt(self) -> int:
        return self.max_sequence_length

    @property
    def n_joint(self) -> int:
        return self.n_txt + self.n_img


def image_position_ids(geo: Flux2Geometry) -> torch.Tensor:
    """(n_img, 4) int64 ids = (T=0, h, w, L=0), token-major (h outer, w inner)."""
    h = torch.arange(geo.token_h)
    w = torch.arange(geo.token_w)
    hh, ww = torch.meshgrid(h, w, indexing="ij")
    ids = torch.zeros(geo.n_img, 4, dtype=torch.int64)
    ids[:, 1] = hh.reshape(-1)
    ids[:, 2] = ww.reshape(-1)
    return ids


def text_position_ids(n_txt: int) -> torch.Tensor:
    """(n_txt, 4) int64 ids = (0, 0, 0, L)."""
    ids = torch.zeros(n_txt, 4, dtype=torch.int64)
    ids[:, 3] = torch.arange(n_txt)
    return ids


def rope_cos_sin(ids: torch.Tensor, axes_dims=(32, 32, 32, 32), theta: float = 2000.0,
                 freqs_dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """diffusers ``Flux2PosEmbed``: per axis ``get_1d_rotary_pos_embed(dim, pos,
    theta, repeat_interleave_real=True, use_real=True)``; returns float32
    (S, sum(axes_dims)) cos / sin in *interleaved pair* layout
    (cos[2j] == cos[2j+1])."""
    cos_out, sin_out = [], []
    for i, dim in enumerate(axes_dims):
        pos = ids[:, i].to(freqs_dtype)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim))
        ang = torch.outer(pos, freqs)                         # (S, dim/2)
        cos_out.append(ang.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(ang.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def apply_rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers ``apply_rotary_emb(..., use_real=True, use_real_unbind_dim=-1)``
    on x of shape (B, S, H, D) with cos/sin (S, D)."""
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return (x.float() * c + x_rot.float() * s).to(x.dtype)


def rotate_half_permutation(head_dim: int = 128) -> torch.Tensor:
    """Index permutation that turns interleaved-pair RoPE into HF rotate-half
    RoPE: new[k] = old[2k], new[k + D/2] = old[2k + 1]. Applied to the Q and K
    projection *rows* (output features) per head and to the q/k RMSNorm gammas,
    attention is unchanged (q.k is invariant to a common permutation)."""
    half = head_dim // 2
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])[:half * 2]


def rope_table_rotate_half(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Convert interleaved (S, D) cos/sin into the engine's rotate-half table
    row ``[cos_h | cos_h | -sin_h | sin_h]`` (S, 2D) bf16, where the pair
    frequencies are gathered in permuted (even-first) order."""
    D = cos.shape[1]
    cos_h = cos[:, 0::2]
    sin_h = sin[:, 0::2]
    assert cos_h.shape[1] == D // 2
    return torch.cat([cos_h, cos_h, -sin_h, sin_h], dim=1).to(torch.bfloat16)


# ----------------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Verbatim from diffusers pipeline_flux2_klein.compute_empirical_mu."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def flow_match_sigmas(image_seq_len: int, num_steps: int) -> torch.Tensor:
    """FlowMatchEulerDiscreteScheduler.set_timesteps with
    use_dynamic_shifting=True / time_shift_type='exponential' and the Klein
    pipeline's ``sigmas = linspace(1, 1/N, N)``. Returns float32 (N+1,) sigmas
    (terminal 0 appended); ``timesteps = sigmas[:-1] * 1000``."""
    import numpy as np
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = compute_empirical_mu(image_seq_len, num_steps)
    sigmas = torch.from_numpy(sigmas).to(torch.float32)
    sigmas = math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0) ** 1.0)
    return torch.cat([sigmas, torch.zeros(1)])


def timestep_sinusoid(t_bf16: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`` then
    ``.to(bf16)`` -- i.e. [cos | sin] of t * exp(-ln(1e4) * i / 128)."""
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    emb = t_bf16.float()[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)   # flip_sin_to_cos
    return emb.to(torch.bfloat16)


def dit_timestep_bf16(sigma: float) -> torch.Tensor:
    """The transformer sees ``timestep = (sigma*1000)/1000`` as float32 then
    casts to bf16 and multiplies by 1000 in bf16."""
    t = torch.tensor([sigma * 1000.0], dtype=torch.float32) / 1000.0
    return t.to(torch.bfloat16) * 1000


# ----------------------------------------------------------------------------
# Latent packing (diffusers Flux2KleinPipeline statics)
# ----------------------------------------------------------------------------

def patchify_latents(lat: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*4, H/2, W/2); channel = c*4 + py*2 + px."""
    B, C, H, W = lat.shape
    x = lat.reshape(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * 4, H // 2, W // 2)


def unpatchify_latents(lat: torch.Tensor) -> torch.Tensor:
    B, C4, h, w = lat.shape
    C = C4 // 4
    x = lat.reshape(B, C, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C, h * 2, w * 2)


def pack_latents(lat: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) -> (B, h*w, C)."""
    B, C, h, w = lat.shape
    return lat.reshape(B, C, h * w).transpose(1, 2)


def unpack_latents(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    B, N, C = tokens.shape
    assert N == h * w
    return tokens.transpose(1, 2).reshape(B, C, h, w)


def initial_noise(geo: Flux2Geometry, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    """``randn_tensor((1, 128, token_h, token_w), generator)`` on a CPU
    generator in the model dtype, then packed to (1, n_img, 128)."""
    gen = torch.Generator("cpu").manual_seed(seed)
    noise = torch.randn((1, geo.in_channels, geo.token_h, geo.token_w),
                        generator=gen, dtype=dtype)
    return pack_latents(noise)


# ----------------------------------------------------------------------------
# Text encoder
# ----------------------------------------------------------------------------

class Qwen3TextEncoder:
    """Qwen3-4B hidden-state taps, exactly as ``Flux2KleinPipeline._get_qwen3_prompt_embeds``."""

    def __init__(self, snap: str, device: str = "cpu", dtype=torch.bfloat16,
                 num_layers: int | None = None):
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
        kw = {}
        if num_layers is not None:
            # HF replaces the LAST entry of ``hidden_states`` with the post-norm
            # output, so to read the *raw* output of layer N (hidden_states[N])
            # the truncated model must keep N+1 layers. The FPGA runs exactly N
            # layers and taps the raw residual stream.
            kw["num_hidden_layers"] = num_layers + 1
        self.model = Qwen3ForCausalLM.from_pretrained(
            os.path.join(snap, "text_encoder"), dtype=dtype, **kw).to(device).eval()
        self.device = device
        self.dtype = dtype

    def tokenize(self, prompt: str, max_sequence_length: int):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = self.tokenizer(text, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=max_sequence_length)
        return enc["input_ids"], enc["attention_mask"]

    @torch.no_grad()
    def __call__(self, prompt: str, max_sequence_length: int = 512,
                 tap_layers=(9, 18, 27)) -> torch.Tensor:
        input_ids, attention_mask = self.tokenize(prompt, max_sequence_length)
        out = self.model(input_ids=input_ids.to(self.device),
                         attention_mask=attention_mask.to(self.device),
                         output_hidden_states=True, use_cache=False)
        hs = torch.stack([out.hidden_states[k] for k in tap_layers], dim=1)  # (B, 3, L, 2560)
        B, nL, L, Hd = hs.shape
        return hs.permute(0, 2, 1, 3).reshape(B, L, nL * Hd).to(self.dtype).cpu()


# ----------------------------------------------------------------------------
# Transformer
# ----------------------------------------------------------------------------

def _linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w.to(x.device))


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """diffusers RMSNorm: fp32 variance, cast to weight dtype, multiply."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    h = x * torch.rsqrt(var + eps)
    return h.to(weight.dtype) * weight.to(h.device)


def _layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return F.silu(x[..., :half]) * x[..., half:]


class Flux2Transformer:
    """Op-for-op re-implementation of diffusers ``Flux2Transformer2DModel``
    (klein-4B config). ``state`` is the raw safetensors dict."""

    def __init__(self, state: dict[str, torch.Tensor], geo: Flux2Geometry,
                 n_double: int = 5, n_single: int = 20, device: str = "cpu"):
        self.s = state
        self.geo = geo
        self.n_double = n_double
        self.n_single = n_single
        self.device = device

    def w(self, key: str) -> torch.Tensor:
        return self.s[key]

    # -- conditioning ------------------------------------------------------
    @torch.no_grad()
    def temb(self, sigma: float) -> torch.Tensor:
        """(1, 3072) bf16 timestep embedding."""
        t = dit_timestep_bf16(sigma)
        w1 = self.w("time_guidance_embed.timestep_embedder.linear_1.weight")
        # the pipeline casts the fp32 sinusoid to the model dtype (bf16) before linear_1
        sincos = timestep_sinusoid(t, self.geo.time_channels).to(device=self.device, dtype=w1.dtype)
        h = _linear(sincos, w1)
        h = F.silu(h)
        return _linear(h, self.w("time_guidance_embed.timestep_embedder.linear_2.weight"))

    @torch.no_grad()
    def modulation(self, temb: torch.Tensor) -> dict[str, torch.Tensor]:
        a = F.silu(temb)
        D = self.geo.hidden
        def chunks(key, n):
            m = _linear(a, self.w(key))
            return [m[:, i * D:(i + 1) * D] for i in range(n)]
        return {
            "img": chunks("double_stream_modulation_img.linear.weight", 6),
            "txt": chunks("double_stream_modulation_txt.linear.weight", 6),
            "single": chunks("single_stream_modulation.linear.weight", 3),
            "out": chunks("norm_out.linear.weight", 2),       # (scale, shift)
        }

    # -- attention ---------------------------------------------------------
    def _attention(self, q, k, v, cos, sin):
        """q/k/v (B, S, 3072) -> (B, S, 3072) with per-head QK RMSNorm already applied."""
        B, S, _ = q.shape
        H, D = self.geo.heads, self.geo.head_dim
        q = q.view(B, S, H, D)
        k = k.view(B, S, H, D)
        v = v.view(B, S, H, D)
        q = apply_rope_interleaved(q, cos, sin)
        k = apply_rope_interleaved(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).reshape(B, S, H * D)

    def _qk_norm(self, x, key):
        B, S, _ = x.shape
        H, D = self.geo.heads, self.geo.head_dim
        return _rms_norm(x.view(B, S, H, D), self.w(key), self.geo.eps).view(B, S, H * D)

    # -- blocks ------------------------------------------------------------
    @torch.no_grad()
    def double_block(self, i, img, txt, mod, cos, sin):
        p = f"transformer_blocks.{i}."
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = mod["img"]
        (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) = mod["txt"]
        eps = self.geo.eps

        x = _layer_norm(img, eps) * (1 + scale_msa)[:, None, :] + shift_msa[:, None, :]
        c = _layer_norm(txt, eps) * (1 + c_scale_msa)[:, None, :] + c_shift_msa[:, None, :]

        q = self._qk_norm(_linear(x, self.w(p + "attn.to_q.weight")), p + "attn.norm_q.weight")
        k = self._qk_norm(_linear(x, self.w(p + "attn.to_k.weight")), p + "attn.norm_k.weight")
        v = _linear(x, self.w(p + "attn.to_v.weight"))
        cq = self._qk_norm(_linear(c, self.w(p + "attn.add_q_proj.weight")), p + "attn.norm_added_q.weight")
        ck = self._qk_norm(_linear(c, self.w(p + "attn.add_k_proj.weight")), p + "attn.norm_added_k.weight")
        cv = _linear(c, self.w(p + "attn.add_v_proj.weight"))

        n_txt = txt.shape[1]
        o = self._attention(torch.cat([cq, q], 1), torch.cat([ck, k], 1), torch.cat([cv, v], 1), cos, sin)
        o_txt, o_img = o[:, :n_txt], o[:, n_txt:]
        attn_img = _linear(o_img, self.w(p + "attn.to_out.0.weight"))
        attn_txt = _linear(o_txt, self.w(p + "attn.to_add_out.weight"))

        img = img + gate_msa[:, None, :] * attn_img
        x2 = _layer_norm(img, eps) * (1 + scale_mlp)[:, None, :] + shift_mlp[:, None, :]
        ff = _linear(_swiglu(_linear(x2, self.w(p + "ff.linear_in.weight"))), self.w(p + "ff.linear_out.weight"))
        img = img + gate_mlp[:, None, :] * ff

        txt = txt + c_gate_msa[:, None, :] * attn_txt
        c2 = _layer_norm(txt, eps) * (1 + c_scale_mlp)[:, None, :] + c_shift_mlp[:, None, :]
        ffc = _linear(_swiglu(_linear(c2, self.w(p + "ff_context.linear_in.weight"))),
                      self.w(p + "ff_context.linear_out.weight"))
        txt = txt + c_gate_mlp[:, None, :] * ffc
        return img, txt

    @torch.no_grad()
    def single_block(self, i, h, mod, cos, sin):
        p = f"single_transformer_blocks.{i}."
        shift, scale, gate = mod["single"]
        D, Dm = self.geo.hidden, self.geo.mlp_hidden
        x = _layer_norm(h, self.geo.eps) * (1 + scale)[:, None, :] + shift[:, None, :]
        proj = _linear(x, self.w(p + "attn.to_qkv_mlp_proj.weight"))
        qkv, mlp = torch.split(proj, [3 * D, 2 * Dm], dim=-1)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._qk_norm(q, p + "attn.norm_q.weight")
        k = self._qk_norm(k, p + "attn.norm_k.weight")
        attn = self._attention(q, k, v, cos, sin)
        mlp = _swiglu(mlp)
        y = _linear(torch.cat([attn, mlp], dim=-1), self.w(p + "attn.to_out.weight"))
        return h + gate[:, None, :] * y

    @torch.no_grad()
    def forward(self, latents: torch.Tensor, prompt_embeds: torch.Tensor, sigma: float,
                cos: torch.Tensor, sin: torch.Tensor, temb: torch.Tensor | None = None,
                mod: dict | None = None, trace: dict | None = None) -> torch.Tensor:
        """latents (1, n_img, 128), prompt_embeds (1, n_txt, 7680) -> velocity (1, n_img, 128)."""
        dev = self.device
        latents = latents.to(dev)
        prompt_embeds = prompt_embeds.to(dev)
        if temb is None:
            temb = self.temb(sigma)
        if mod is None:
            mod = self.modulation(temb)
        img = _linear(latents, self.w("x_embedder.weight"))
        txt = _linear(prompt_embeds, self.w("context_embedder.weight"))
        cos = cos.to(dev)
        sin = sin.to(dev)
        if trace is not None:
            trace["temb"], trace["mod"] = temb.cpu(), {k: [t.cpu() for t in v] for k, v in mod.items()}
            trace["img_embed"], trace["txt_embed"] = img.cpu(), txt.cpu()
        for i in range(self.n_double):
            img, txt = self.double_block(i, img, txt, mod, cos, sin)
            if trace is not None:
                trace[f"double_{i}"] = torch.cat([txt, img], dim=1).cpu()
        h = torch.cat([txt, img], dim=1)
        for i in range(self.n_single):
            h = self.single_block(i, h, mod, cos, sin)
            if trace is not None:
                trace[f"single_{i}"] = h.cpu()
        h = h[:, txt.shape[1]:]
        scale, shift = mod["out"]
        h = _layer_norm(h, self.geo.eps) * (1 + scale)[:, None, :] + shift[:, None, :]
        if trace is not None:
            trace["norm_out"] = h.cpu()
        return _linear(h, self.w("proj_out.weight")).cpu()


def denoise(dit: Flux2Transformer, geo: Flux2Geometry, prompt_embeds: torch.Tensor,
            latents: torch.Tensor, sigmas: torch.Tensor | None = None,
            callback=None) -> torch.Tensor:
    """Euler flow-matching loop (scheduler.step in fp32, result back to bf16)."""
    ids = torch.cat([text_position_ids(prompt_embeds.shape[1]), image_position_ids(geo)], 0)
    cos, sin = rope_cos_sin(ids, geo.axes_dims_rope, geo.rope_theta)
    if sigmas is None:
        sigmas = flow_match_sigmas(geo.n_img, geo.num_inference_steps)
    x = latents
    for s in range(len(sigmas) - 1):
        v = dit.forward(x, prompt_embeds, float(sigmas[s]), cos, sin)
        dt = sigmas[s + 1] - sigmas[s]
        x = (x.float() + dt * v.float()).to(latents.dtype)
        if callback is not None:
            callback(s, float(sigmas[s]), v, x)
    return x


# ----------------------------------------------------------------------------
# VAE decoder
# ----------------------------------------------------------------------------

class Flux2VaeDecoder:
    """AutoencoderKLFlux2 decode path: BN de-normalisation of the packed
    latents, unpatchify, post_quant_conv, Decoder(conv_in, mid[res, attn, res],
    4 up levels x 3 resnets (+ nearest-2x upsample conv), GroupNorm/SiLU/conv_out)."""

    def __init__(self, state: dict[str, torch.Tensor], cfg: dict, device: str = "cpu",
                 dtype=torch.float32):
        self.s = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v
                  for k, v in state.items()}
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.groups = cfg.get("norm_num_groups", 32)
        self.block_out = list(cfg.get("block_out_channels", [128, 256, 512, 512]))
        self.layers_per_block = cfg.get("layers_per_block", 2) + 1
        self.bn_eps = cfg.get("batch_norm_eps", 1e-4)

    def w(self, k):
        return self.s[k]

    def conv(self, x, key, pad):
        return F.conv2d(x, self.w(key + ".weight"), self.w(key + ".bias"), padding=pad)

    def gn(self, x, key, eps=1e-6):
        return F.group_norm(x, self.groups, self.w(key + ".weight"), self.w(key + ".bias"), eps)

    def resnet(self, x, p, cin, cout):
        h = self.conv(F.silu(self.gn(x, p + ".norm1")), p + ".conv1", 1)
        h = self.conv(F.silu(self.gn(h, p + ".norm2")), p + ".conv2", 1)
        if cin != cout:
            x = self.conv(x, p + ".conv_shortcut", 0)
        return x + h

    def attn(self, x, p):
        B, C, H, W = x.shape
        h = self.gn(x, p + ".group_norm")
        h = h.view(B, C, H * W).transpose(1, 2)                       # (B, HW, C)
        q = F.linear(h, self.w(p + ".to_q.weight"), self.w(p + ".to_q.bias"))
        k = F.linear(h, self.w(p + ".to_k.weight"), self.w(p + ".to_k.bias"))
        v = F.linear(h, self.w(p + ".to_v.weight"), self.w(p + ".to_v.bias"))
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        o = F.linear(o, self.w(p + ".to_out.0.weight"), self.w(p + ".to_out.0.bias"))
        return x + o.transpose(1, 2).reshape(B, C, H, W)

    def denormalize(self, packed: torch.Tensor, geo: Flux2Geometry) -> torch.Tensor:
        """(1, n_img, 128) packed tokens -> (1, 32, latent_h, latent_w) latent."""
        lat = unpack_latents(packed, geo.token_h, geo.token_w)
        mean = self.w("bn.running_mean").view(1, -1, 1, 1).to(lat.dtype)
        std = torch.sqrt(self.w("bn.running_var").view(1, -1, 1, 1) + self.bn_eps).to(lat.dtype)
        lat = lat.to(mean.device) * std + mean
        return unpatchify_latents(lat)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """(1, 32, h, w) latent -> (1, 3, 8h, 8w) image in [-1, 1]."""
        x = latent.to(device=self.device, dtype=self.dtype)
        x = self.conv(x, "post_quant_conv", 0)
        x = self.conv(x, "decoder.conv_in", 1)
        c = self.block_out[-1]
        x = self.resnet(x, "decoder.mid_block.resnets.0", c, c)
        x = self.attn(x, "decoder.mid_block.attentions.0")
        x = self.resnet(x, "decoder.mid_block.resnets.1", c, c)
        rev = list(reversed(self.block_out))
        for i, cout in enumerate(rev):
            for j in range(self.layers_per_block):
                cin = c if j == 0 else cout
                x = self.resnet(x, f"decoder.up_blocks.{i}.resnets.{j}", cin, cout)
            c = cout
            if i < len(rev) - 1:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
                x = self.conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv", 1)
        x = F.silu(self.gn(x, "decoder.conv_norm_out"))
        return self.conv(x, "decoder.conv_out", 1).cpu()


def to_uint8_image(img: torch.Tensor):
    """(1, 3, H, W) in [-1, 1] -> (H, W, 3) uint8 (diffusers postprocess)."""
    x = (img[0].float() / 2 + 0.5).clamp(0, 1)
    return (x * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous()


# ----------------------------------------------------------------------------
# Fake quantization (simulate the FPGA weight codebooks on the host)
# ----------------------------------------------------------------------------

def fake_quantize_tensor(w: torch.Tensor, precision: str = "if4", block_size: int = 64) -> torch.Tensor:
    """Round-trip a 2-D (N, K) weight through quant_lib's packer+unpacker."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    import quant_lib
    N, K = w.shape
    data, scale = quant_lib.quantize(precision, w.to(torch.bfloat16).cpu(), block_size=block_size)
    return quant_lib.dequant(precision, data, scale, N, K, block_size=block_size).to(w.dtype)


def fake_quantize_state(state: dict[str, torch.Tensor], keys, precision: str = "if4",
                        block_size: int = 64) -> dict[str, torch.Tensor]:
    """Return a copy of ``state`` with the listed 2-D weights fake-quantized."""
    out = dict(state)
    for k in keys:
        out[k] = fake_quantize_tensor(state[k], precision, block_size)
    return out


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

@dataclass
class ReferenceResult:
    prompt_embeds: torch.Tensor
    noise: torch.Tensor
    latents: torch.Tensor
    image: torch.Tensor                      # (H, W, 3) uint8
    sigmas: torch.Tensor
    steps: list = field(default_factory=list)


def run_reference(prompt: str, geo: Flux2Geometry, seed: int = 0, *, snap: str | None = None,
                  device: str = "cpu", transformer_state=None, vae_state=None,
                  text_encoder: Qwen3TextEncoder | None = None,
                  prompt_embeds: torch.Tensor | None = None,
                  keep_steps: bool = False) -> ReferenceResult:
    snap = snap or snapshot_dir()
    if prompt_embeds is None:
        te = text_encoder or Qwen3TextEncoder(snap, device=device, num_layers=geo.text_layers_used)
        prompt_embeds = te(prompt, geo.max_sequence_length, geo.text_tap_layers)
        if text_encoder is None:
            del te
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    if transformer_state is None:
        transformer_state = load_safetensors_dir(os.path.join(snap, "transformer"), device=device)
    dit = Flux2Transformer(transformer_state, geo, device=device)
    noise = initial_noise(geo, seed)
    sigmas = flow_match_sigmas(geo.n_img, geo.num_inference_steps)
    steps = []
    cb = (lambda s, sg, v, x: steps.append((s, sg, v.clone(), x.clone()))) if keep_steps else None
    latents = denoise(dit, geo, prompt_embeds, noise, sigmas, callback=cb)
    del dit, transformer_state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if vae_state is None:
        vae_state = load_safetensors_dir(os.path.join(snap, "vae"))
    vae = Flux2VaeDecoder(vae_state, load_json(os.path.join(snap, "vae", "config.json")), device=device)
    image = to_uint8_image(vae.decode(vae.denormalize(latents, geo)))
    return ReferenceResult(prompt_embeds, noise, latents, image, sigmas, steps)


def save_image(img_u8: torch.Tensor, path: str) -> None:
    from PIL import Image
    Image.fromarray(img_u8.numpy()).save(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FLUX.2 klein 4B torch reference")
    ap.add_argument("--prompt", default="A photo of a red sports car on a mountain road at sunset")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="flux2_ref.png")
    args = ap.parse_args()
    geo = Flux2Geometry(args.height, args.width, args.max_seq_len, args.steps)
    res = run_reference(args.prompt, geo, args.seed, device=args.device)
    save_image(res.image, args.out)
    print(f"saved {args.out}; sigmas={res.sigmas.tolist()}")
