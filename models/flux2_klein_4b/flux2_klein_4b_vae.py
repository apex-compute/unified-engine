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

import user_dma_core as udc  # noqa: E402
from user_dma_core import TYPE, UE_VECTOR_SIZE  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402


def conv2d_layer_sharded(engines, x: torch.Tensor, w_codes: torch.Tensor, *, pad: int,
                         block_scales: torch.Tensor, bias: torch.Tensor, data_type=TYPE.IF8,
                         timeout_s: float = 300.0) -> torch.Tensor:
    """``UnifiedEngine.run_conv2d_layer`` spread over several engines.

    The layer driver already splits a conv into output-channel chunks and
    same-geometry tile groups; every (chunk, group) tile range is divided
    evenly across the engines, each engine captures its own tile-loop program
    over its share (same staged activation, same per-chunk weight streams,
    disjoint output tiles), all programs are launched together and the host
    waits for every engine. One staging DMA and one readback, as before."""
    ue0 = engines[0]
    ne = len(engines)
    assert data_type in (TYPE.IF4, TYPE.IF8)
    C, H, W_ = x.shape
    OC, wc_c, kh, kw = w_codes.shape
    assert wc_c == C
    stride_s, dilation, pad_h = 1, 1, pad
    ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
    taps = kh * kw * ct
    lalu_mode, lalu_a, lalu_b = udc._conv_fused_lalu(False, False, False)
    patch_taps = kh * kw * C
    chunks = (patch_taps + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
    can_gather = C <= 255 and patch_taps <= 256
    gather_cycles = max(kh * kw * ((C + 3) // 4), OC * chunks)
    use_gather = can_gather and gather_cycles < OC * taps
    block_scales = block_scales.detach().to(device="cpu", dtype=torch.bfloat16)
    assert tuple(block_scales.shape) == (OC, chunks if use_gather else taps)
    out_h, out_w, oc_chunk, tiles = udc.plan_conv2d_layer_tiles(
        c_in=C, oc_count=OC, in_h=H, in_w=W_, kernel_h=kh, kernel_w=kw, stride_s=stride_s,
        pad=pad, pad_h=pad_h, dilation=dilation, gather=use_gather, bias_enabled=bias is not None)
    n_chunks = OC // oc_chunk
    groups, activation_bytes, chunk_output_bytes = udc.conv2d_tiled_dram_layout(tiles, ct, oc_chunk)
    staged = udc.conv2d_pack_activation_tiles(x, tiles, pad=pad, pad_h=pad_h)
    ACT_BASE = ue0.allocate_params_dram(activation_bytes)
    ue0.dma_write(ue0.h2c_device, ACT_BASE, staged, staged.numel() * 2)
    OUT_BASE = ue0.allocate_tensor_dram(n_chunks * chunk_output_bytes)
    stream_group = max(groups, key=lambda g: g[2] * g[3])
    sth, stw = stream_group[2], stream_group[3]
    streams = []
    for ci, oc0 in enumerate(range(0, OC, oc_chunk)):
        wc = w_codes[oc0:oc0 + oc_chunk]
        chunk_scale = udc._conv2d_chunk_scale(block_scales, 1.0, oc0=oc0, n_oc=oc_chunk)
        if use_gather:
            w_stream = udc.conv2d_pack_weight_stream_gather(wc, sth, stw, data_type)
            scale_stream = udc.conv2d_pack_scale_stream_gather(chunk_scale, oc_chunk, chunks)
        else:
            w_stream = udc.conv2d_pack_weight_stream(wc, sth, stw, data_type)
            scale_stream = udc.conv2d_pack_scale_stream(chunk_scale, oc_chunk, taps, sth, stw)
        WA = ue0.allocate_params_dram(w_stream.numel())
        ue0.dma_write(ue0.h2c_device, WA, w_stream, w_stream.numel())
        SA = ue0.allocate_params_dram(scale_stream.numel() * 2)
        ue0.dma_write(ue0.h2c_device, SA, scale_stream, scale_stream.numel() * 2)
        BA = None
        if bias is not None:
            bias_stream = udc.conv2d_pack_bias_stream(bias[oc0:oc0 + oc_chunk], sth, stw)
            BA = ue0.allocate_params_dram(bias_stream.numel() * 2)
            ue0.dma_write(ue0.h2c_device, BA, bias_stream, bias_stream.numel() * 2)
        streams.append((WA, SA, BA))
    # work units: every (chunk, geometry group) tile range split evenly across engines
    per_engine = [[] for _ in engines]
    for ci in range(n_chunks):
        for gi, g in enumerate(groups):
            start, stop = g[0], g[1]
            n = stop - start
            base, rem = divmod(n, ne)
            t = start
            for e in range(ne):
                cnt = base + (1 if e < rem else 0)
                if cnt:
                    per_engine[e].append((ci, gi, t, t + cnt))
                    t += cnt
    progs = []
    for e, eng in enumerate(engines):
        if not per_engine[e]:
            progs.append(None)
            continue
        eng.clear_inst_id()
        eng.reset_isa_reg_counter()
        eng.reset_inst_ptr_counter()
        eng.start_capture()
        for ci, gi, t0, t1 in per_engine[e]:
            (start, stop, th, tw, _wh, win_w, act_offset, act_bytes, out_offset, out_bytes, _rl) = groups[gi]
            rel = t0 - start
            WA, SA, BA = streams[ci]
            eng._capture_conv2d_tile_loop(
                n_tiles=t1 - t0,
                act_base=ACT_BASE + act_offset + rel * act_bytes, act_bytes=act_bytes,
                weights_dram_addr=WA,
                out_base=OUT_BASE + ci * chunk_output_bytes + out_offset + rel * out_bytes, out_bytes=out_bytes,
                kernel_w=kw, kernel_h=kh, ct=ct, oc_count=oc_chunk, out_w=tw, out_h=th,
                w_pad=win_w, stride_s=stride_s, dilation=dilation, data_type=data_type,
                scale_dram_addr=SA, scale_count=(oc_chunk * chunks if use_gather else oc_chunk * taps),
                bias_dram_addr=BA, results=th * tw * oc_chunk,
                lalu_mode=lalu_mode, lalu_a=lalu_a, lalu_b=lalu_b,
                gather=use_gather, c_in=C)
        eng.stop_capture()
        eng.generate_instruction_halt()
        prog = eng.get_program_dram_addr()
        eng.write_captured_instructions_to_dram(prog)
        eng.allocate_program_dram(eng.get_capture_instruction_size_bytes())
        eng.clear_capture_buffer()
        progs.append(prog)
    for eng, prog in zip(engines, progs):
        if prog is not None:
            eng.start_execute_from_dram(prog)
    for eng, prog in zip(engines, progs):
        if prog is not None:
            eng.wait_queue(timeout_s)
    big = ue0.dma_from_accelerator_memory(OUT_BASE, (n_chunks * chunk_output_bytes // 2,))
    return udc.conv2d_unpack_tiled_result(big, tiles, out_h, out_w, OC, oc_chunk)


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
        # the engines that run the convolutions (all of them) and where each
        # one's per-layer programs go (its cursor after the DiT phase)
        self.engines = list(getattr(ue, "E", [ue]))
        self.prog_bases = [eng.get_program_dram_addr() for eng in self.engines]

    # ---- arena discipline: every layer starts from the same marks ----
    def _reset(self) -> None:
        ue = self.ue
        ue.reset_params_dram_addr()
        ue._tensor_dram_addr = getattr(ue, "VAE_TENSOR_BASE", ue._tensor_dram_base)
        for eng, base in zip(self.engines, self.prog_bases):
            eng._next_program_dram_addr = base

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
        if len(self.engines) > 1:
            return self._timed(key, lambda: conv2d_layer_sharded(
                self.engines, x, c["codes"], pad=c["pad"], block_scales=c["block_scales"], bias=c["bias"],
                data_type=TYPE.IF8, timeout_s=self.timeout_s))
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
