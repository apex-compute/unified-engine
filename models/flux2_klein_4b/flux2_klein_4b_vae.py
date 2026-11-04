#!/usr/bin/env python3
"""FLUX.2 VAE decoder on the Unified Engine (phase 3).

``AutoencoderKLFlux2`` decode path driven layer-by-layer through the existing
convolution-model drivers in ``user_dma_core``:

  conv (k3/k1, bias, IF8 per-block scales)   run_conv2d_layer
  GroupNorm(32, affine, eps 1e-6)            run_group_norm
  SiLU                                       run_silu_layer
  nearest 2x upsample                        run_nn_upsample_2x (then conv3x3)
  residual add                               run_eltwise_add_layer
  mid-block single-head attention            unified_attention_core (seq = h*w, head_dim = C)

Every arithmetic op runs on the FPGA; the host only moves (C, H, W) maps
between layers (that is how every conv model in this repo executes today).
The DRAM arenas are reset before every layer (U-Net backend pattern), so the
decoder's footprint is one layer's staging regardless of resolution.

Needs the CONV2D / MAXPOOL queue-config FPGA image. The host reference with
the same IF8 weights (``reference_decode``) is the oracle for this phase.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from user_dma_core import TYPE, UE_VECTOR_SIZE  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402


class Flux2VaeFPGA:
    def __init__(self, ue, weights: dict, cfg: dict, *, progress: bool = True, timeout_s: float = 900.0):
        self.ue = ue
        self.w = weights
        self.cfg = cfg["vae"]
        self.groups = self.cfg["norm_num_groups"]
        self.gn_eps = self.cfg.get("gn_eps", 1e-6)
        self.block_out = list(self.cfg["block_out_channels"])
        self.n_res = self.cfg["layers_per_block"] + 1
        self.progress = progress
        self.timeout_s = timeout_s
        self.layer_times: list[tuple[str, float]] = []

    # ---- arena discipline: every layer starts from the same marks ----
    def _reset(self) -> None:
        ue = self.ue
        ue.reset_params_dram_addr()
        ue._tensor_dram_addr = getattr(ue, "VAE_TENSOR_BASE", ue._tensor_dram_base)
        ue._next_program_dram_addr = getattr(ue, "VAE_PROGRAM_BASE", ue._program_dram_base)

    def _timed(self, name, fn):
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        self.layer_times.append((name, dt))
        if self.progress:
            print(f"    [vae] {name:52s} {tuple(out.shape)} {dt:6.1f}s", flush=True)
        return out

    # ---- primitives ----
    def conv(self, x: torch.Tensor, key: str) -> torch.Tensor:
        c = self.w["convs"][key]
        self._reset()
        return self._timed(key, lambda: self.ue.run_conv2d_layer(
            x, c["codes"], stride_s=1, pad=c["pad"], block_scales=c["block_scales"], bias=c["bias"],
            data_type=TYPE.IF8, timeout_s=self.timeout_s))

    def gn(self, x: torch.Tensor, key: str) -> torch.Tensor:
        n = self.w["norms"][key]
        self._reset()
        return self._timed(key, lambda: self.ue.run_group_norm(
            x, num_groups=self.groups, gamma=n["gamma"], beta=n["beta"], eps=self.gn_eps, timeout_s=self.timeout_s))

    def silu(self, x: torch.Tensor, key: str) -> torch.Tensor:
        self._reset()
        return self._timed(key + ".silu", lambda: self.ue.run_silu_layer(x, timeout_s=self.timeout_s))

    def add(self, x: torch.Tensor, y: torch.Tensor, key: str) -> torch.Tensor:
        self._reset()
        return self._timed(key + ".residual", lambda: self.ue.run_eltwise_add_layer(x, y, timeout_s=self.timeout_s))

    def upsample(self, x: torch.Tensor, key: str) -> torch.Tensor:
        self._reset()
        return self._timed(key + ".nearest2x", lambda: self.ue.run_nn_upsample_2x(x, timeout_s=self.timeout_s))

    def attention(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """x (C, H, W): GN -> q/k/v (1x1 conv + bias) -> softmax(q k^T / sqrt(C)) v -> out 1x1 -> + x."""
        C, H, Wd = x.shape
        seq = H * Wd
        assert C % UE_VECTOR_SIZE == 0 and seq % UE_VECTOR_SIZE == 0
        h = self.gn(x, p + ".group_norm")
        q = self.conv(h, p + ".to_q")
        k = self.conv(h, p + ".to_k")
        v = self.conv(h, p + ".to_v")
        ue = self.ue
        self._reset()
        bpe = 2
        Q, K, V = (ue.allocate_tensor_dram(seq * C * bpe) for _ in range(3))
        BIAS = ue.allocate_tensor_dram(seq * seq * bpe)
        OUT = ue.allocate_tensor_dram(seq * C * bpe)
        SCRATCH = ue.allocate_tensor_dram((C * seq + seq * seq + seq * C) * bpe)
        IDENT = ue.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe)
        t0 = time.time()
        ue.start_capture()
        ue.unified_attention_core(batch=seq, aligned_seq_len=seq, head_dim=C,
                                  Q_DRAM_ADDR=Q, K_DRAM_ADDR=K, V_DRAM_ADDR=V, BIAS_DRAM_ADDR=BIAS,
                                  OUTPUT_DRAM_ADDR=OUT, SCRATCH_DRAM_ADDR=SCRATCH, IDENTITY_DRAM_ADDR=IDENT)
        ue.stop_capture()
        ue.generate_instruction_halt()
        prog = ue.get_program_dram_addr()
        nbytes = ue.get_capture_instruction_size_bytes()
        ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(nbytes)
        for addr, t in ((Q, q), (K, k), (V, v)):
            ue.dma_to_accelerator_memory(addr, t.reshape(C, seq).t().contiguous())   # (seq, C) rows
        ue.dma_to_accelerator_memory(BIAS, torch.zeros(seq, seq, dtype=torch.bfloat16))
        ue.dma_to_accelerator_memory(IDENT, torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))
        ue.start_execute_from_dram(prog)
        ue.wait_queue(self.timeout_s)
        ue.clear_capture_buffer()
        attn = ue.dma_from_accelerator_memory(OUT, (seq, C))
        self.layer_times.append((p + ".attention", time.time() - t0))
        if self.progress:
            print(f"    [vae] {p + '.attention':52s} ({seq}, {C}) {time.time() - t0:6.1f}s", flush=True)
        o = self.conv(attn.t().reshape(C, H, Wd).contiguous(), p + ".to_out.0")
        return self.add(x, o, p)

    def resnet(self, x: torch.Tensor, p: str, cin: int, cout: int) -> torch.Tensor:
        h = self.conv(self.silu(self.gn(x, p + ".norm1"), p + ".norm1"), p + ".conv1")
        h = self.conv(self.silu(self.gn(h, p + ".norm2"), p + ".norm2"), p + ".conv2")
        if cin != cout:
            x = self.conv(x, p + ".conv_shortcut")
        return self.add(x, h, p)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """(32, h, w) bf16 latent (already BN de-normalised) -> (3, 8h, 8w) bf16 in [-1, 1]."""
        x = latent.to(torch.bfloat16)
        x = self.conv(x, "post_quant_conv")
        x = self.conv(x, "decoder.conv_in")
        c = self.block_out[-1]
        x = self.resnet(x, "decoder.mid_block.resnets.0", c, c)
        x = self.attention(x, "decoder.mid_block.attentions.0")
        x = self.resnet(x, "decoder.mid_block.resnets.1", c, c)
        rev = list(reversed(self.block_out))
        for i, cout in enumerate(rev):
            for j in range(self.n_res):
                x = self.resnet(x, f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
            c = cout
            if i < len(rev) - 1:
                x = self.upsample(x, f"decoder.up_blocks.{i}.upsamplers.0")
                x = self.conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv")
        x = self.silu(self.gn(x, "decoder.conv_norm_out"), "decoder.conv_norm_out")
        return self.conv(x, "decoder.conv_out")


# ----------------------------------------------------------------------------
# Host oracle with the same IF8 weights
# ----------------------------------------------------------------------------

def reference_decode(weights: dict, cfg: dict, latent: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Same graph on the host from the dequantized IF8 codes (bias in bf16)."""
    v = cfg["vae"]
    groups, eps = v["norm_num_groups"], v.get("gn_eps", 1e-6)
    convs, norms = weights["convs"], weights["norms"]
    x = latent[None].to(dtype)

    def conv(x, key):
        c = convs[key]
        w = W.dequantize_conv_if8(c["codes"], c["block_scales"]).to(dtype)
        return F.conv2d(x, w, c["bias"].to(dtype), padding=c["pad"])

    def gn(x, key):
        n = norms[key]
        return F.group_norm(x, groups, n["gamma"].to(dtype), n["beta"].to(dtype), eps)

    def resnet(x, p, cin, cout):
        h = conv(F.silu(gn(x, p + ".norm1")), p + ".conv1")
        h = conv(F.silu(gn(h, p + ".norm2")), p + ".conv2")
        if cin != cout:
            x = conv(x, p + ".conv_shortcut")
        return x + h

    def attn(x, p):
        B, C, H, Wd = x.shape
        h = gn(x, p + ".group_norm")
        q, k, vv = (conv(h, p + f".{n}").reshape(B, C, H * Wd).transpose(1, 2) for n in ("to_q", "to_k", "to_v"))
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], vv[:, None])[:, 0]
        o = conv(o.transpose(1, 2).reshape(B, C, H, Wd), p + ".to_out.0")
        return x + o

    x = conv(x, "post_quant_conv")
    x = conv(x, "decoder.conv_in")
    c = v["block_out_channels"][-1]
    x = resnet(x, "decoder.mid_block.resnets.0", c, c)
    x = attn(x, "decoder.mid_block.attentions.0")
    x = resnet(x, "decoder.mid_block.resnets.1", c, c)
    rev = list(reversed(v["block_out_channels"]))
    for i, cout in enumerate(rev):
        for j in range(v["layers_per_block"] + 1):
            x = resnet(x, f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
        c = cout
        if i < len(rev) - 1:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv")
    x = F.silu(gn(x, "decoder.conv_norm_out"))
    return conv(x, "decoder.conv_out")[0]


def decoder_cost(cfg: dict, latent_h: int, latent_w: int) -> dict:
    """MACs per primitive for a (32, latent_h, latent_w) decode (host estimate)."""
    v = cfg["vae"]
    rev = list(reversed(v["block_out_channels"]))
    macs = 0
    ops = 0
    def conv(cin, cout, h, w, k):
        nonlocal macs, ops
        macs += h * w * cout * k * k * cin
        ops += 1
    h, w = latent_h, latent_w
    c = rev[0]
    conv(32, 32, h, w, 1); conv(32, c, h, w, 3)
    def resnet(cin, cout):
        conv(cin, cout, h, w, 3); conv(cout, cout, h, w, 3)
        if cin != cout:
            conv(cin, cout, h, w, 1)
    resnet(c, c)
    for _ in range(4):
        conv(c, c, h, w, 1)
    macs += 2 * (h * w) ** 2 * c
    resnet(c, c)
    for i, cout in enumerate(rev):
        for j in range(v["layers_per_block"] + 1):
            resnet(c if j == 0 else cout, cout)
        c = cout
        if i < len(rev) - 1:
            h, w = 2 * h, 2 * w
            conv(c, c, h, w, 3)
    conv(c, 3, h, w, 3)
    return {"macs": macs, "convs": ops, "out": (3, h, w)}
