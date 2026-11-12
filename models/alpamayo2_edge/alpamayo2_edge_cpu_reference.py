#!/usr/bin/env python3
"""Alpamayo 2 Edge host reference (CPU / CUDA) in plain PyTorch -- no transformers.

nvidia/Alpamayo2-Edge, 2.65B params, bf16 checkpoint. Stages:

  vision     SigLIP2 ViT, 27 layers x 1152, 16 heads x 72, patch 16, attention per image
  projector  LayerNorm(1152) -> 2x2 merge (4608) -> 11520 -> GELU -> 2048
  prefill    Cosmos3-Edge LM, 28 layers x 2048, GQA 16q/8kv x 128, relu^2 MLP 9216,
             RMSNorm, interleaved M-RoPE (theta 1e8), causal
  decode     reasoning text, one token at a time, until <|traj_future_start|>
  expert     28 layers x 512, same KV geometry: 64 action tokens attend non-causally to
             [LM KV cache ; themselves]; 10 Euler flow-matching steps
  traj       64 x (accel, curvature) -> unicycle integration -> xyz / yaw   (host math)

Inputs are the packed tensors of NVIDIA's processor; utility/capture_golden.py writes
them together with stage goldens from NVIDIA's implementation.

  # plumbing only (random inputs, no golden needed)
  python models/alpamayo2_edge/alpamayo2_edge_cpu_reference.py --selfcheck
  # parity against the golden: teacher-forces the reasoning tokens and the initial noise
  python models/alpamayo2_edge/alpamayo2_edge_cpu_reference.py --golden ~/a2e_golden/<sample>
  # free-running (greedy reasoning, own noise); ADE against the logged future if present
  python models/alpamayo2_edge/alpamayo2_edge_cpu_reference.py --golden ~/a2e_golden/<sample> --decode greedy
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

HF_REPO = "nvidia/Alpamayo2-Edge"
HF_REVISION = "f8951d095de3589b3face1dec3f4394c8ecedb38"

VIS = "vlm.model.visual."
PROJ = "vlm.model.projector."
LM = "vlm.model.language_model."
EXP = "expert.expert."
AIP = "expert.action_in_proj."
STAT_KEYS = ("accel_mean", "accel_std", "curvature_mean", "curvature_std")


# ----------------------------------------------------------------------------- primitives

def layer_norm(x, w, b, eps):
    return F.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), eps).to(x.dtype)


def rms_norm(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def attention(q, k, v, scale, mask=None):
    """q (H, Lq, D), k / v (H, Lk, D) -> (H, Lq, D). Softmax in float32."""
    a = (torch.matmul(q, k.transpose(-1, -2)) * scale).float()
    if mask is not None:
        a = a + mask
    return torch.matmul(torch.softmax(a, dim=-1).to(q.dtype), v)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin


class KVCache:
    """Rotated keys / values of every LM layer: (layers, kv_heads, capacity, head_dim)."""

    def __init__(self, layers, heads, capacity, head_dim, device, dtype):
        self.K = torch.empty(layers, heads, capacity, head_dim, device=device, dtype=dtype)
        self.V = torch.empty_like(self.K)
        self.n = 0

    def past(self, layer):
        return (self.K[layer, :, : self.n], self.V[layer, :, : self.n]) if self.n else None

    def put(self, layer, k, v):
        self.K[layer, :, self.n : self.n + k.shape[1]] = k
        self.V[layer, :, self.n : self.n + v.shape[1]] = v


# ----------------------------------------------------------------------------- model

class Alpamayo2EdgeRef:
    def __init__(self, model_dir, device="cpu", dtype=torch.float32, prefixes=None):
        """prefixes: load only the weights under these name prefixes (e.g. (VIS,) for the vision tower)."""
        self.dir, self.dev, self.dt = Path(model_dir), torch.device(device), dtype
        cfg = json.loads((self.dir / "config.json").read_text())
        vlm, exp = cfg["vlm_config"], cfg["expert_config"]
        text, vis = vlm["text_config"], vlm["vision_config"]
        for key in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"):
            assert text[key] == exp["llm_config"][key], f"LM / expert KV geometry differs: {key}"
        self.n_lm, self.n_vis = text["num_hidden_layers"], vis["num_hidden_layers"]
        self.q_heads, self.kv_heads, self.hd = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"]
        self.rms_eps = text["rms_norm_eps"]
        self.theta = text["rope_parameters"]["rope_theta"]
        self.sections = text["rope_parameters"]["mrope_section"]
        self.vis_heads, self.vis_eps, self.merge = vis["num_attention_heads"], vis["layer_norm_eps"], vis["spatial_merge_size"]
        self.image_token, self.vision_start, self.vision_end = vlm["image_token_id"], vlm["vision_start_token_id"], vlm["vision_end_token_id"]
        self.eos = text["eos_token_id"]
        ids = cfg["traj_ids"]
        self.anchor = ids["future_start"]
        self.masked = [(ids["history_id0"], cfg["history_vocab_size"]), (ids["future_id0"], cfg["future_vocab_size"])]
        self.asp = exp["action_space_cfg"]
        self.n_way = self.asp["n_waypoints"]
        aip = exp["action_in_proj_cfg"]
        # NVIDIA builds these non-persistent buffers under the checkpoint dtype, so its bf16 model
        # runs with bf16-rounded Fourier frequencies / action statistics (60.0 instead of 59.948).
        # Mirror that per dtype; a golden overrides both with the values NVIDIA really used.
        self.freqs = torch.logspace(0, math.log10(aip["max_freq"]), aip["num_fourier_feats"] // 2, dtype=dtype).float().to(self.dev)
        self.stats = {k: torch.tensor(self.asp[k], dtype=dtype).float().item() for k in STAT_KEYS}

        self.w = {}
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            tensors = load_file(str(self.dir / shard))
            for name in list(tensors):
                t = tensors.pop(name)
                if prefixes is None or name.startswith(tuple(prefixes)):
                    self.w[name] = t.to(self.dev, dtype)

    # ------------------------------------------------------------------ vision + projector

    def vision_pos_embed(self, grid_thw):
        """Learned 16x16 grid, bilinearly resized per image, in the processor's 2x2 block-major order."""
        pe = self.w[VIS + "embeddings.position_embedding.weight"].float()
        side = math.isqrt(pe.shape[0])
        pe = pe.reshape(side, side, -1).permute(2, 0, 1)[None]
        m, out, memo = self.merge, [], {}
        for t, h, wd in grid_thw.tolist():
            if (h, wd) not in memo:
                r = F.interpolate(pe, size=(h, wd), mode="bilinear", align_corners=False, antialias=True)
                r = r[0].permute(1, 2, 0).reshape(h // m, m, wd // m, m, -1)
                memo[(h, wd)] = r.transpose(1, 2).reshape(h * wd, -1).to(self.dt)
            out.append(memo[(h, wd)].repeat(t, 1))
        return torch.cat(out)

    def vision_encode(self, pixel_values, grid_thw, stages=None, hw_gelu=False, layer_outs=None):
        """pixel_values (N, 768) normalized patches of all images, packed -> (N, 1152).

        hw_gelu: use the accelerator's x*sigmoid(1.702x) in place of tanh-GELU, so the result
        predicts the hardware instead of the published model. layer_outs: list that receives the
        residual stream after every layer."""
        w = self.w
        x = F.linear(pixel_values.to(self.dev, self.dt), w[VIS + "embeddings.patch_embedding.weight"], w[VIS + "embeddings.patch_embedding.bias"])
        x = x + self.vision_pos_embed(grid_thw)
        if stages is not None:
            stages["vision_embed"] = x
        assert all(t == 1 for t, _, _ in grid_thw.tolist()), "images only (temporal patch size 1)"
        sizes = [h * wd for _, h, wd in grid_thw.tolist()]
        H = self.vis_heads
        D = x.shape[-1] // H
        for layer in range(self.n_vis):
            p = f"{VIS}encoder.layers.{layer}."
            h = layer_norm(x, w[p + "layer_norm1.weight"], w[p + "layer_norm1.bias"], self.vis_eps)
            q = F.linear(h, w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"])
            k = F.linear(h, w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"])
            v = F.linear(h, w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"])
            outs, s = [], 0
            for n in sizes:  # every image is its own attention sequence
                qi, ki, vi = (a[s : s + n].view(n, H, D).transpose(0, 1) for a in (q, k, v))
                outs.append(attention(qi, ki, vi, D ** -0.5).transpose(0, 1).reshape(n, H * D))
                s += n
            x = x + F.linear(torch.cat(outs), w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"])
            h = layer_norm(x, w[p + "layer_norm2.weight"], w[p + "layer_norm2.bias"], self.vis_eps)
            h = F.linear(h, w[p + "mlp.fc1.weight"], w[p + "mlp.fc1.bias"])
            h = h * torch.sigmoid(1.702 * h) if hw_gelu else F.gelu(h, approximate="tanh")
            x = x + F.linear(h, w[p + "mlp.fc2.weight"], w[p + "mlp.fc2.bias"])
            if layer_outs is not None:
                layer_outs.append(x)
        return layer_norm(x, w[VIS + "post_layernorm.weight"], w[VIS + "post_layernorm.bias"], self.vis_eps)

    def project(self, vis, hw_gelu=False):
        """(N, 1152) -> (N / 4, 2048): the four patches of a 2x2 block are adjacent rows."""
        w = self.w
        h = layer_norm(vis, w[PROJ + "norm.weight"], w[PROJ + "norm.bias"], 1e-6)
        h = h.reshape(-1, self.merge ** 2 * vis.shape[-1])
        h = F.linear(h, w[PROJ + "linear_fc1.weight"], w[PROJ + "linear_fc1.bias"])
        h = h * torch.sigmoid(1.702 * h) if hw_gelu else F.gelu(h)
        return F.linear(h, w[PROJ + "linear_fc2.weight"], w[PROJ + "linear_fc2.bias"])

    # ------------------------------------------------------------------ positions

    def mrope_positions(self, mm_types, grid_thw):
        """(t, h, w) position of every prompt token and the text offset `delta`.

        Text advances all three axes together. An image holds t at its start position and
        spans h / w over its merged grid, then advances by max(grid_h, grid_w). Tokens after
        the last image -- generated text, action tokens -- sit at (sequence index + delta).
        """
        types = mm_types.tolist()
        L = len(types)
        pos = torch.zeros(3, L, dtype=torch.long)
        grids = iter(grid_thw.tolist())
        cur = i = 0
        while i < L:
            j = i
            while j < L and types[j] == types[i]:
                j += 1
            if types[i] == 0:
                pos[:, i:j] = torch.arange(j - i) + cur
                cur += j - i
            else:
                t, h, wd = next(grids)
                gh, gw = h // self.merge, wd // self.merge
                assert j - i == t * gh * gw, "image placeholder count does not match its grid"
                pos[0, i:j] = cur
                pos[1, i:j] = (torch.arange(gh) + cur).view(gh, 1).expand(gh, gw).reshape(-1)
                pos[2, i:j] = (torch.arange(gw) + cur).view(1, gw).expand(gh, gw).reshape(-1)
                cur += max(h, wd) // self.merge
            i = j
        return pos, int(pos.max()) + 1 - L

    def rope_cos_sin(self, pos3):
        """pos3 (3, L) -> cos, sin (L, head_dim). Frequency slots 1,4,..,58 take h, 2,5,..,59 take w, the rest t."""
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, dtype=torch.float) / self.hd))
        freqs = pos3.float()[:, :, None] * inv
        f = freqs[0].clone()
        for axis in (1, 2):
            slot = slice(axis, self.sections[axis] * 3, 3)
            f[:, slot] = freqs[axis][:, slot]
        emb = torch.cat([f, f], -1)
        return emb.cos().to(self.dev, self.dt), emb.sin().to(self.dev, self.dt)

    # ------------------------------------------------------------------ LM / expert layer

    def block(self, prefix, layer, x, cos, sin, past=None, mask=None):
        """One decoder layer of the LM (2048 wide) or the expert (512 wide). x (L, hidden).

        past = (K, V), each (kv_heads, P, head_dim), keys already rotated.
        Returns (x_out, k, v): k / v are this call's rotated keys and values.
        """
        w, p, L = self.w, f"{prefix}layers.{layer}.", x.shape[0]
        h = rms_norm(x, w[p + "input_layernorm.weight"], self.rms_eps)
        q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(L, self.q_heads, self.hd).transpose(0, 1)
        k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(L, self.kv_heads, self.hd).transpose(0, 1)
        v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(L, self.kv_heads, self.hd).transpose(0, 1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        K, V = (k, v) if past is None else (torch.cat([past[0], k], 1), torch.cat([past[1], v], 1))
        rep = self.q_heads // self.kv_heads
        o = attention(q, K.repeat_interleave(rep, 0), V.repeat_interleave(rep, 0), self.hd ** -0.5, mask)
        x = x + F.linear(o.transpose(0, 1).reshape(L, -1), w[p + "self_attn.o_proj.weight"])
        h = rms_norm(x, w[p + "post_attention_layernorm.weight"], self.rms_eps)
        h = F.relu(F.linear(h, w[p + "mlp.fc1.weight"])).square()
        return x + F.linear(h, w[p + "mlp.fc2.weight"]), k, v

    def lm_forward(self, embeds, pos3, cache, stages=None, layer_outs=None):
        """Causal pass over new tokens (L, 2048) at pos3 (3, L); appends their KV. Returns final-norm hidden."""
        cos, sin = self.rope_cos_sin(pos3)
        L, P = embeds.shape[0], cache.n
        mask = torch.full((L, P + L), float("-inf"), device=self.dev).triu(P + 1) if L > 1 else None
        x = embeds
        for layer in range(self.n_lm):
            x, k, v = self.block(LM, layer, x, cos, sin, cache.past(layer), mask)
            cache.put(layer, k, v)
            if stages is not None and layer in (0, self.n_lm - 1):
                stages[f"prefill_layer{layer}"] = x
            if layer_outs is not None:
                layer_outs.append(x)
        cache.n += L
        return rms_norm(x, self.w[LM + "norm.weight"], self.rms_eps)

    def embed_prompt(self, input_ids, image_embeds):
        e = self.w[LM + "embed_tokens.weight"][input_ids.to(self.dev)]
        is_image = (input_ids == self.image_token).to(self.dev)
        assert int(is_image.sum()) == image_embeds.shape[0], "image tokens != projected patches"
        e[is_image] = image_embeds.to(e.dtype)
        return e

    # ------------------------------------------------------------------ reasoning decode

    def pick(self, logits, mode, gen, temperature, top_p):
        logits = logits.clone()
        for start, count in self.masked:  # discrete trajectory codes never appear in the text
            logits[start : start + count] = float("-inf")
        if self.eos != self.anchor:  # the text may only end on the anchor
            logits[self.eos] = float("-inf")
        if mode == "greedy":
            return int(logits.argmax())
        logits = logits / temperature
        ordered, order = torch.sort(logits)
        drop = ordered.softmax(-1).cumsum(-1) <= 1.0 - top_p
        drop[-1] = False
        logits[order[drop]] = float("-inf")
        return int(torch.multinomial(logits.softmax(-1), 1, generator=gen))

    def decode(self, last_token, index, delta, cache, mode="greedy", teacher=None, max_new=256,
               gen=None, temperature=0.6, top_p=0.98, stages=None):
        """Generate until the anchor. The anchor is fed through the LM too, so its KV is cached.

        `last_token` is the final prompt token (not prefilled) at sequence index `index`.
        """
        emb, head = self.w[LM + "embed_tokens.weight"], self.w["vlm.lm_head.weight"]
        tok, out, top_ids, top_vals = int(last_token), [], [], []
        while True:
            h = self.lm_forward(emb[tok][None], torch.full((3, 1), index + delta), cache)
            if out and out[-1] == self.anchor:
                break
            if len(out) >= max_new or (mode == "teacher" and len(out) >= len(teacher)):
                raise RuntimeError(f"no <|traj_future_start|> within {len(out)} generated tokens")
            logits = F.linear(h[0], head).float()
            if stages is not None:
                if not out:
                    stages["step_logits0"] = logits
                vals, idx = logits.topk(5)
                top_ids.append(idx)
                top_vals.append(vals)
            out.append(int(teacher[len(out)]) if mode == "teacher" else self.pick(logits, mode, gen, temperature, top_p))
            tok, index = out[-1], index + 1
        if stages is not None:
            stages["step_top5_ids"], stages["step_top5_vals"] = torch.stack(top_ids), torch.stack(top_vals)
        return out

    # ------------------------------------------------------------------ action expert

    def fourier(self, u):
        arg = u.float()[..., None] * self.freqs * 2 * math.pi
        return torch.cat([arg.sin(), arg.cos()], -1) * math.sqrt(2)

    def action_in_proj(self, x, t):
        """x (64, 2) noisy normalized (accel, curvature), t scalar in [0, 1] -> (64, 512)."""
        w, n = self.w, x.shape[0]
        f = torch.cat([self.fourier(x[:, 0]), self.fourier(x[:, 1]), self.fourier(t).expand(n, -1)], -1).to(self.dt)
        h = F.silu(F.linear(f, w[AIP + "encoder.trunk.0.weight"], w[AIP + "encoder.trunk.0.bias"]))
        h = rms_norm(h, w[AIP + "encoder.trunk.2.weight"], 1e-5)
        h = F.silu(F.linear(h, w[AIP + "encoder.trunk.3.weight"], w[AIP + "encoder.trunk.3.bias"]))
        h = rms_norm(h, w[AIP + "encoder.trunk.5.weight"], 1e-5)
        h = F.linear(h, w[AIP + "encoder.trunk.6.weight"], w[AIP + "encoder.trunk.6.bias"])
        return layer_norm(h, w[AIP + "norm.weight"], w[AIP + "norm.bias"], 1e-5)

    def denoise(self, cache, delta, x0, steps=10, stages=None):
        """Euler-integrate the flow field from noise x0 (64, 2), conditioned on the cached prefix."""
        n = self.n_way
        pos = (cache.n + delta + torch.arange(n)).expand(3, n)
        cos, sin = self.rope_cos_sin(pos)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.dev)
        x = x0.to(self.dev).float()
        trace = {"x_steps": [], "expert_in": [], "expert_hidden": [], "v_steps": []}
        for i in range(steps):
            h = self.action_in_proj(x, ts[i])
            trace["x_steps"].append(x)
            trace["expert_in"].append(h)
            for layer in range(self.n_lm):  # the action tokens' own KV is never cached
                h, _, _ = self.block(EXP, layer, h, cos, sin, cache.past(layer))
            h = rms_norm(h, self.w[EXP + "norm.weight"], self.rms_eps)
            v = F.linear(h, self.w["expert.action_out_proj.weight"], self.w["expert.action_out_proj.bias"]).float()
            trace["expert_hidden"].append(h)
            trace["v_steps"].append(v)
            x = x + (ts[i + 1] - ts[i]) * v
        if stages is not None:
            stages["expert_pos"] = pos
            stages.update({key: torch.stack(val) for key, val in trace.items()})
        return x

    # ------------------------------------------------------------------ action -> trajectory

    def t0_states(self, xyz, rot):
        """Speed and course at t0 from the 20 history poses (float32, as NVIDIA solves it)."""
        a = self.asp
        dt = a["dt"]
        yaw_raw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        yaw = _unwrap(yaw_raw)
        dxy = xyz[1:, :2] - xyz[:-1, :2]
        N = dxy.shape[0]
        r = torch.arange(N)
        A = torch.zeros(2 * N, N + 1)  # trapezoid rule: v_t u_t + v_t+1 u_t+1 = 2 dp_t / dt
        A[2 * r, r], A[2 * r, r + 1] = yaw.cos()[:-1], yaw.cos()[1:]
        A[2 * r + 1, r], A[2 * r + 1, r + 1] = yaw.sin()[:-1], yaw.sin()[1:]
        D = torch.zeros(N - 2, N + 1)  # third-difference smoothness
        q = torch.arange(N - 2)
        D[q, q], D[q, q + 1], D[q, q + 2], D[q, q + 3] = -1.0, 3.0, -3.0, 1.0
        lhs = A.T @ A + a["v_lambda"] / dt ** 6 * (D.T @ D) + a["v_ridge"] * torch.eye(N + 1)
        rhs = A.T @ (2 / dt * dxy).reshape(-1)
        v = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(lhs))[:, 0]
        theta0 = _course_offset(xyz[-1, :2] - xyz[-2, :2], _unwrap(yaw_raw[-2:]).mean(), dt)
        return v[-1], theta0

    def action_to_traj(self, action, hist_xyz, hist_rot):
        """action (64, 2) normalized -> xyz (64, 3), rot (64, 3, 3) in the t0 ego frame."""
        a, st = self.asp, self.stats
        dt = a["dt"]
        action, xyz, rot = action.float().cpu(), hist_xyz.float().cpu(), hist_rot.float().cpu()
        acc = action[:, 0] * st["accel_std"] + st["accel_mean"]
        kap = action[:, 1] * st["curvature_std"] + st["curvature_mean"]
        v0, th0 = self.t0_states(xyz, rot)
        v = torch.cat([v0[None], v0 + (acc * dt).cumsum(0)])
        th = torch.cat([th0[None], th0 + (kap * v[:-1] * dt).cumsum(0) + (kap * acc * 0.5 * dt ** 2).cumsum(0)])
        half = 0.5 * dt
        x = (v[:-1] * th[:-1].cos() * half).cumsum(0) + (v[1:] * th[1:].cos() * half).cumsum(0)
        y = (v[:-1] * th[:-1].sin() * half).cumsum(0) + (v[1:] * th[1:].sin() * half).cumsum(0)
        lateral = v * torch.gradient(th, spacing=dt)[0]
        yaw = (th - th0 + a["slip_gradient"] * (lateral - lateral[0]))[1:]  # body yaw = course + sideslip
        c, s, zero, one = yaw.cos(), yaw.sin(), torch.zeros_like(yaw), torch.ones_like(yaw)
        out_rot = torch.stack([torch.stack([c, -s, zero], -1), torch.stack([s, c, zero], -1), torch.stack([zero, zero, one], -1)], -2)
        return torch.stack([x, y, xyz[-1, 2].expand_as(x)], -1), out_rot


def _unwrap(phi):
    d = torch.diff(phi)
    return torch.cat([phi[:1], phi[:1] + torch.atan2(d.sin(), d.cos()).cumsum(0)])


def _course_offset(chord, yaw, dt):
    """Course minus body yaw along a position chord; 0 beyond 50 mrad, faded in over 0.5..1 m/s."""
    c, s = yaw.cos(), yaw.sin()
    along, across = c * chord[0] + s * chord[1], c * chord[1] - s * chord[0]
    off = torch.atan2(across * torch.sign(along), along.abs())
    off = torch.where(off.abs() > 0.05, torch.zeros_like(off), off)
    return off * ((chord.norm() / dt - 0.5) / 0.5).clamp(0.0, 1.0)


# ----------------------------------------------------------------------------- driver

def snr_db(ref, x):
    ref, x = ref.double().cpu(), x.double().cpu()
    noise = (ref - x).pow(2).sum()
    return float("inf") if noise == 0 else 10 * math.log10(float(ref.pow(2).sum() / noise))


def cos_sim(ref, x):
    return float(F.cosine_similarity(ref.double().cpu().flatten(), x.double().cpu().flatten(), dim=0))


def ade(pred, gt, steps):
    return float((pred[:steps, :2] - gt[:steps, :2]).norm(dim=-1).mean())


def resolve_model_dir(path):
    if path:
        return Path(path).expanduser()
    from huggingface_hub import snapshot_download

    want = dict(repo_id=HF_REPO, revision=HF_REVISION, allow_patterns=["*.json", "*.safetensors"])
    try:
        return Path(snapshot_download(local_files_only=True, **want))
    except Exception:
        return Path(snapshot_download(**want))


def synthetic_inputs(m):
    """Two tiny images in a short random prompt and a straight 10 m/s history."""
    g = torch.Generator().manual_seed(0)
    grid = torch.tensor([[1, 4, 6], [1, 4, 4]])
    ids, types = [1] + torch.randint(1000, 5000, (5,), generator=g).tolist(), [0] * 6
    for t, h, wd in grid.tolist():
        n = t * h * wd // m.merge ** 2
        ids += [m.vision_start] + [m.image_token] * n + [m.vision_end] + torch.randint(1000, 5000, (3,), generator=g).tolist()
        types += [0] + [1] * n + [0] * 4
    xyz = torch.zeros(20, 3)
    xyz[:, 0] = torch.arange(-19.0, 1.0)
    return {
        "tokenized_data.input_ids": torch.tensor(ids)[None],
        "tokenized_data.mm_token_type_ids": torch.tensor(types)[None],
        "tokenized_data.image_grid_thw": grid,
        "tokenized_data.pixel_values": torch.randn(int(grid.prod(-1).sum()), 768, generator=g),
        "traj_data.ego_history_xyz": xyz[None, None],
        "traj_data.ego_history_rot": torch.eye(3).expand(1, 1, 20, 3, 3).contiguous(),
    }


def run(m, inp, args, golden=None):
    """Full pipeline on one packed sample. Returns (outputs, stages of the first sample, timings)."""
    ids = inp["tokenized_data.input_ids"][0]
    types = inp["tokenized_data.mm_token_type_ids"][0]
    grid = inp["tokenized_data.image_grid_thw"]
    hist_xyz = inp["traj_data.ego_history_xyz"].reshape(-1, 3)
    hist_rot = inp["traj_data.ego_history_rot"].reshape(-1, 3, 3)
    if "tokenized_data.attention_mask" in inp:
        assert bool(inp["tokenized_data.attention_mask"].all()), "padded prompts are not supported"
    L, stages, clock = ids.shape[0], {}, {}

    def timed(name, fn):
        t = time.perf_counter()
        out = fn()
        if m.dev.type == "cuda":
            torch.cuda.synchronize()
        clock[name] = clock.get(name, 0.0) + time.perf_counter() - t
        return out

    vis = timed("vision", lambda: m.vision_encode(inp["tokenized_data.pixel_values"], grid, stages, hw_gelu=args.hw_gelu))
    stages["vision_out"] = vis
    img = timed("projector", lambda: m.project(vis, hw_gelu=args.hw_gelu))
    stages["image_embeds"] = img
    pos, delta = m.mrope_positions(types, grid)
    stages["prefill_pos"] = pos[:, : L - 1]
    embeds = m.embed_prompt(ids, img)
    cache = KVCache(m.n_lm, m.kv_heads, L + args.max_new + 1, m.hd, m.dev, m.dt)
    stages["prefill_hidden"] = timed("prefill", lambda: m.lm_forward(embeds[: L - 1], pos[:, : L - 1], cache, stages))
    prefilled = cache.n

    mode = args.decode or ("teacher" if golden is not None else "greedy")
    teacher = golden["generated_ids"].tolist() if mode == "teacher" else None
    out = {"pred_xyz": [], "pred_rot": [], "action": [], "generated_ids": []}
    for sample in range(args.num_samples):
        cache.n = prefilled  # every candidate continues from the shared prompt
        st = stages if sample == 0 else None
        gen = torch.Generator(device=m.dev).manual_seed(args.seed + sample)
        toks = timed("decode", lambda: m.decode(ids[-1], L - 1, delta, cache, mode, teacher, args.max_new, gen, stages=st))
        if st is not None:
            for layer in (0, m.n_lm - 1):
                st[f"kv_k{layer}"], st[f"kv_v{layer}"] = cache.K[layer, :, : cache.n], cache.V[layer, :, : cache.n]
        if golden is not None and mode == "teacher" and sample == 0:
            x0 = golden["x_steps"][0]
        else:
            x0 = torch.randn(m.n_way, 2, generator=torch.Generator().manual_seed(args.seed + sample))
        action = timed("expert", lambda: m.denoise(cache, delta, x0, args.steps, st))
        xyz, rot = m.action_to_traj(action, hist_xyz, hist_rot)
        for key, val in (("pred_xyz", xyz), ("pred_rot", rot), ("action", action.cpu()), ("generated_ids", torch.tensor(toks))):
            out[key].append(val)
    clock["tokens"] = (L, len(out["generated_ids"][0]))
    return out, stages, clock


def compare(stages, out, golden):
    """Stage-by-stage SNR / cosine against NVIDIA's implementation."""
    stages = dict(stages, action=out["action"][0], pred_xyz=out["pred_xyz"][0], pred_rot=out["pred_rot"][0])
    print(f"{'stage':<16}{'shape':<20}{'SNR dB':>9}{'cos':>10}{'max|err|':>11}")
    for name in ("vision_embed", "vision_out", "image_embeds", "prefill_layer0", "prefill_hidden",
                 "kv_k0", "kv_v0", "step_logits0", "expert_in", "expert_hidden", "v_steps", "x_steps",
                 "action", "pred_xyz", "pred_rot"):
        names = [name] if name in golden else []
        if name in ("prefill_layer0", "kv_k0", "kv_v0"):  # also the last layer
            names += [n for n in golden if n.startswith(name[:-1]) and n != name]
        for n in names:
            if n not in stages:
                continue
            ref, x = golden[n].float(), stages[n].float().cpu()
            if ref.shape != x.shape:
                print(f"{n:<16}shape mismatch: golden {tuple(ref.shape)} vs {tuple(x.shape)}")
                continue
            print(f"{n:<16}{str(tuple(x.shape)):<20}{snr_db(ref, x):9.1f}{cos_sim(ref, x):10.6f}{float((ref - x).abs().max()):11.3g}")
    for name in ("prefill_pos", "expert_pos"):
        if name in golden:
            same = torch.equal(golden[name].long(), stages[name].long().cpu())
            print(f"{name:<16}{'exact' if same else 'MISMATCH'}")
    if "step_top5_ids" in golden:
        n = min(golden["step_top5_ids"].shape[0], stages["step_top5_ids"].shape[0])
        hit = (golden["step_top5_ids"][:n, 0] == stages["step_top5_ids"][:n, 0].cpu()).sum()
        gap = (golden["step_top5_vals"][:n] - stages["step_top5_vals"][:n].cpu()).abs().max()
        print(f"{'decode argmax':<16}{int(hit)}/{n} steps agree, top-5 logit max|err| {float(gap):.3g}")
    v_snr = [snr_db(golden["v_steps"][i], stages["v_steps"][i]) for i in range(min(len(golden["v_steps"]), len(stages["v_steps"])))]
    print("v per step dB   " + " ".join(f"{s:.0f}" for s in v_snr))


def selfcheck(m, args):
    args.max_new, args.steps, args.decode, args.num_samples = 8, 2, "teacher", 1
    inp = synthetic_inputs(m)
    golden = {"generated_ids": torch.tensor([2000, 2001, 2002, m.anchor]), "x_steps": torch.randn(1, m.n_way, 2)}
    out, stages, clock = run(m, inp, args, golden)
    for key in ("vision_out", "image_embeds", "prefill_hidden", "v_steps"):
        assert torch.isfinite(stages[key].float()).all(), f"{key} is not finite"
    assert out["pred_xyz"][0].shape == (m.n_way, 3) and out["pred_rot"][0].shape == (m.n_way, 3, 3)
    assert torch.isfinite(out["pred_xyz"][0]).all() and torch.isfinite(out["pred_rot"][0]).all()
    # the KV cache path must equal one causal pass over prompt + generated tokens
    ids = torch.cat([inp["tokenized_data.input_ids"][0], golden["generated_ids"]])
    pos, delta = m.mrope_positions(inp["tokenized_data.mm_token_type_ids"][0], inp["tokenized_data.image_grid_thw"])
    L = pos.shape[1]
    pos = torch.cat([pos, (torch.arange(L, len(ids)) + delta).expand(3, -1)], 1)
    embeds = m.w[LM + "embed_tokens.weight"][ids.to(m.dev)]
    embeds[: L] = m.embed_prompt(ids[:L], stages["image_embeds"])
    full = KVCache(m.n_lm, m.kv_heads, len(ids), m.hd, m.dev, m.dt)
    m.lm_forward(embeds, pos, full)
    last = m.n_lm - 1
    snr = snr_db(full.K[last], stages[f"kv_k{last}"])
    print(f"selfcheck: shapes / finite ok, incremental vs one-shot last-layer keys {snr:.1f} dB, "
          f"x[63]={out['pred_xyz'][0][-1, 0]:.2f} m, timings {({k: round(v, 2) for k, v in clock.items() if k != 'tokens'})}")
    assert snr > (60 if m.dt == torch.float32 else 20), "KV cache path diverges from the one-shot pass"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="checkpoint directory (default: HF cache snapshot)")
    ap.add_argument("--golden", help="capture_golden.py output dir: packed.safetensors [+ golden.safetensors]")
    ap.add_argument("--inputs", help="packed.safetensors, when there is no golden dir")
    ap.add_argument("--decode", choices=["teacher", "greedy", "sample"], help="default: teacher with a golden, else greedy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", help="write predictions to this .safetensors")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--hw-gelu", action="store_true", help="vision + projector GELU = x*sigmoid(1.702x), the accelerator's activation")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    t = time.perf_counter()
    m = Alpamayo2EdgeRef(resolve_model_dir(args.model_dir), args.device, getattr(torch, args.dtype))
    print(f"loaded {len(m.w)} tensors on {args.device}/{args.dtype} in {time.perf_counter() - t:.1f}s")
    if args.selfcheck:
        return selfcheck(m, args)

    gdir = Path(args.golden).expanduser() if args.golden else None
    inp = load_file(str(gdir / "packed.safetensors" if gdir else args.inputs))
    golden = load_file(str(gdir / "golden.safetensors")) if gdir and (gdir / "golden.safetensors").exists() else None
    if golden is not None:  # run with the constants NVIDIA's model really held
        m.freqs = golden["freqs"].float().to(m.dev)
        m.stats = dict(zip(STAT_KEYS, golden["act_stats"].float().tolist()))
    out, stages, clock = run(m, inp, args, golden)

    L, n_gen = clock.pop("tokens")
    print(f"prompt {L} tokens, {n_gen} generated | " + ", ".join(f"{k} {v:.2f}s" for k, v in clock.items()))
    tok_json = m.dir / "tokenizer.json"
    if tok_json.exists():
        from tokenizers import Tokenizer

        print("reasoning:", Tokenizer.from_file(str(tok_json)).decode(out["generated_ids"][0].tolist()[:-1], skip_special_tokens=False).strip())
    if golden is not None and (args.decode or "teacher") == "teacher":
        compare(stages, out, golden)
    if golden is not None and "gt_xyz" in golden:
        gt = golden["gt_xyz"]
        ours = [ade(xyz, gt, 50) for xyz in out["pred_xyz"]]
        print(f"ADE@5s vs logged future: ours min {min(ours):.3f} m over {len(ours)} sample(s), "
              f"NVIDIA {ade(golden['pred_xyz'], gt, 50):.3f} m | ADE@6.4s ours {ade(out['pred_xyz'][0], gt, 64):.3f} m")
    if args.out:
        save_file({k: torch.stack(v).contiguous() if k != "generated_ids" else v[0] for k, v in out.items()}, args.out)


if __name__ == "__main__":
    main()
