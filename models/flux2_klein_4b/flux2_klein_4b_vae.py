#!/usr/bin/env python3
"""FLUX.2 VAE decoder on the Unified Engine (phase 3) as ONE program.

``AutoencoderKLFlux2`` decode path emitted into every engine's stream as the
third phase of the program set (after the text encoder and the DiT), with the
scheduler's flag barriers between ops. The host DMAs the latent map and the
constant image in before the phase and reads the final image out after it;
nothing moves between host and board while it runs.

On-device layout. Every activation map is stored channels-in-lanes
(``conv2d_pack_activation_map``: pixel-major, ``ct = ceil(C/64)`` 128-byte
lines per pixel) with a one-pixel zero border ("padded"); the producers keep
the border zero. The mid-block attention operands are the same map without the
border ("compact", a (seq, C) matrix); the final conv writes its tiles
contiguously ("tiled") for the host to unpack.

  1x1 conv    with C and OC multiples of 64 the map IS a row-major [pixels, C]
              matrix: one quantized-B matmul (A @ W^T + bias) per image row.
  conv        the layer driver's exact tile grid; per tile one strided DRAM->URAM
              read of the window straight from the padded map, CONV2D, and strided
              URAM->DRAM writes of the tile rows into the output map. Tile ranges
              of every (OC chunk, geometry group) are split over the engines. The
              walker re-reads its weight stream per tile pixel, so each conv is
              preceded by an on-engine expansion of its unrepeated weight blocks
              (loaded once per run in the constant image).
  GroupNorm   pass 1 lane-wise pixel sums into slotted accumulators (flushed every
              32 chunks); the flush blocks are transposed with an identity matmul
              and summed by a ones matmul on the bf20 ladder; a group-average
              matmul pools per group; pass 2 does the same for the centred squares;
              rsqrt by Newton iteration; pass 3 applies x*A_c + B_c.
  SiLU        the layer driver's max(x,0)/exp/polynomial form; the [x, 0] pair
              map it needs is built on device with a strided write.
  add         rounds of ELTWISE_ADD.
  upsample    per input row: one read, four strided writes (2 rows x 2 parities).
  attention   unified_attention_core over query-row shards.

The host reference with the same IF8 weights (``reference_decode``) is the
oracle; ``reference_decode(..., intermediates=...)`` returns every op's
output so the on-board probe can check any prefix of the graph.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import user_dma_core as udc  # noqa: E402
from user_dma_core import TYPE, UE_VECTOR_SIZE, INSTRUCTION_SIZE_BYTES  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402

LINE = UE_VECTOR_SIZE * 2          # one 64-lane bf16 URAM line (bytes)
URAM_B = 0x80000                   # URAM bank B byte offset
WB = 0x300 << 7                    # conv writeback region in URAM_A (bytes)
NEWTON_ITERS = 40                  # rsqrt from y0 = 2^-10 up to 1e3
MAP_ALIGN = 256 * 1024             # map buffers are padded so round-based ops may overrun
K_PIECE = 8192                     # ladder reduction piece (rows) per matmul
GN_FLUSH = udc.GROUP_NORM_MAX_ACC_DEPTH


def emitter_hash() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def _split(n: int, parts: int, unit: int = 1) -> list[tuple[int, int]]:
    """Even [start, stop) split of ``n`` items (in multiples of ``unit``) over ``parts``."""
    assert n % unit == 0
    blocks = n // unit
    base, rem = divmod(blocks, parts)
    out, t = [], 0
    for e in range(parts):
        cnt = (base + (1 if e < rem else 0)) * unit
        out.append((t, t + cnt))
        t += cnt
    return out


def _i16bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.int16).numpy().tobytes()


def _align(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


class Buf:
    """A DRAM activation map."""
    __slots__ = ("addr", "layout", "C", "H", "W", "ct", "size")

    def __init__(self, addr, layout, C, H, W):
        self.addr, self.layout, self.C, self.H, self.W = addr, layout, C, H, W
        self.ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        self.size = Buf.bytes_of(layout, C, H, W)

    @staticmethod
    def bytes_of(layout, C, H, W):
        ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        if layout == "padded":
            n = (H + 2) * (W + 2) * ct * LINE
        elif layout == "compact":
            n = H * W * ct * LINE
        else:
            raise ValueError(layout)
        return _align(n, MAP_ALIGN)

    @property
    def Hp(self):
        return self.H + 2 if self.layout == "padded" else self.H

    @property
    def Wp(self):
        return self.W + 2 if self.layout == "padded" else self.W

    @property
    def border(self):
        return 1 if self.layout == "padded" else 0

    @property
    def pixels(self):
        return self.Hp * self.Wp

    @property
    def lines(self):
        return self.pixels * self.ct

    def shape(self):
        return (self.C, self.H, self.W)


class Flux2VaeOneShot:
    """Planner (shape-only, once) + per-engine emitter of the VAE phase, and the
    run-time helpers (constant image, latent packing, output unpacking)."""

    def __init__(self, ue, cfg: dict):
        self.ue = ue                                   # Flux2KleinEngine: .E .ne .cur .e .B()
        self.ne = ue.ne
        self.cfg = cfg["vae"]
        self.groups = self.cfg["norm_num_groups"]
        self.gn_eps = self.cfg.get("gn_eps", 1e-6)
        self.block_out = list(self.cfg["block_out_channels"])
        self.n_res = self.cfg["layers_per_block"] + 1
        self.lat = self.cfg["latent_channels"]
        self.steps: list[dict] = []
        self.consts: dict[str, int] = {}
        self.gn_params: dict[str, int] = {}
        self.conv_chunks: dict[str, dict] = {}
        self.macs = 0

    # ------------------------------------------------------------------ allocators
    def _calloc(self, n: int, align: int = 64) -> int:
        self._const = _align(self._const, align)
        a = self._const
        self._const += n
        return a

    def _alloc(self, n: int, align: int = 4096) -> int:
        self._cur = _align(self._cur, align)
        a = self._cur
        self._cur += n
        self.arena_end = max(self.arena_end, self._cur)
        return a

    def _scratch(self, name: str, size: int) -> int:
        """A region reused by every step that asks for ``name`` (sized by the dry pass)."""
        if self._dry:
            self._scratch_max[name] = max(self._scratch_max.get(name, 0), size)
            return 0
        if name not in self._scratch_addr:
            self._scratch_addr[name] = self._alloc(self._scratch_max[name])
        assert size <= self._scratch_max[name]
        return self._scratch_addr[name]

    def _new_buf(self, layout, C, H, W) -> Buf:
        size = Buf.bytes_of(layout, C, H, W)
        pool = self._free.setdefault(size, [])
        addr = pool.pop() if pool else self._alloc(size)
        return Buf(addr, layout, C, H, W)

    def _release(self, b: Buf) -> None:
        self._free.setdefault(b.size, []).append(b.addr)

    # ------------------------------------------------------------------ shapes
    def conv_shapes(self) -> dict[str, tuple[int, int, int, int, int]]:
        """(OC, C, kh, kw, pad) of every decoder conv from the config alone."""
        lat = self.lat
        s = {"post_quant_conv": (lat, lat, 1, 1, 0)}
        c = self.block_out[-1]
        s["decoder.conv_in"] = (c, lat, 3, 3, 1)

        def resnet(p, cin, cout):
            s[p + ".conv1"] = (cout, cin, 3, 3, 1)
            s[p + ".conv2"] = (cout, cout, 3, 3, 1)
            if cin != cout:
                s[p + ".conv_shortcut"] = (cout, cin, 1, 1, 0)
        resnet("decoder.mid_block.resnets.0", c, c)
        for nm in ("to_q", "to_k", "to_v", "to_out.0"):
            s[f"decoder.mid_block.attentions.0.{nm}"] = (c, c, 1, 1, 0)
        resnet("decoder.mid_block.resnets.1", c, c)
        rev = list(reversed(self.block_out))
        for i, cout in enumerate(rev):
            for j in range(self.n_res):
                resnet(f"decoder.up_blocks.{i}.resnets.{j}", c if j == 0 else cout, cout)
            c = cout
            if i < len(rev) - 1:
                s[f"decoder.up_blocks.{i}.upsamplers.0.conv"] = (c, c, 3, 3, 1)
        s["decoder.conv_out"] = (self.cfg.get("out_channels", 3), c, 3, 3, 1)
        return s

    @staticmethod
    def _conv_geometry(C, H, Wd, OC, kh, kw, pad):
        ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        taps = kh * kw * ct
        patch = kh * kw * C
        chunks = (patch + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
        can_gather = C <= 255 and patch <= 256
        use_gather = can_gather and max(kh * kw * ((C + 3) // 4), OC * chunks) < OC * taps
        out_h, out_w, oc_chunk, tiles = udc.plan_conv2d_layer_tiles(
            c_in=C, oc_count=OC, in_h=H, in_w=Wd, kernel_h=kh, kernel_w=kw, stride_s=1,
            pad=pad, pad_h=pad, dilation=1, gather=use_gather, bias_enabled=True)
        groups, act_bytes, chunk_out = udc.conv2d_tiled_dram_layout(tiles, ct, oc_chunk)
        sg = max(groups, key=lambda g: g[2] * g[3])
        return dict(ct=ct, taps=taps, chunks=chunks, use_gather=use_gather, out_h=out_h, out_w=out_w,
                    oc_chunk=oc_chunk, n_chunks=OC // oc_chunk, tiles=tiles, groups=groups,
                    act_bytes=act_bytes, chunk_out=chunk_out, npix=sg[2] * sg[3],
                    blocks_per_oc=chunks if use_gather else taps)

    # ------------------------------------------------------------------ plan
    def plan(self, lat_h: int, lat_w: int) -> dict:
        """Shape-only pass: lays out the constant image and every buffer, records the ops."""
        ue = self.ue
        self.shapes = self.conv_shapes()
        self.steps = []
        self._free = {}
        base = ue._params_dram_base
        self.const_base = self._const = base
        # ---- constants independent of the layers
        C_list = sorted({self.shapes[k][1] for k in self.shapes} | {self.shapes[k][0] for k in self.shapes})
        C_list = [c for c in C_list if c % UE_VECTOR_SIZE == 0]
        self.consts["ZERO"] = self._calloc(udc.URAM_NEAR_FULL_SIZE)                 # DRAM zeros (any URAM load)
        self.consts["ONES"] = self._calloc(64 * 8192 * 2)                           # all-ones [64, K<=8192]
        self.consts["Y0"] = self._calloc(512 * 2)                                    # 2^-10 row (Newton seed)
        for c in C_list:
            self.consts[f"IDENT{c}"] = self._calloc(c * c * 2)
            self.consts[f"GAVG{c}"] = self._calloc(c * c * 2)                       # group-average matrix
        seq = lat_h * lat_w
        self.consts["ATT_BIAS"] = self._calloc(seq * seq * 2)
        if "IDENT64" not in self.consts:
            self.consts["IDENT64"] = self._calloc(64 * 64 * 2)
        fixed_end = self._const
        # ---- dry walk: const image size and the scratch maxima
        self._scratch_max, self._scratch_addr = {}, {}
        self._cur = self.arena_end = 0
        self._dry = True
        self._walk(lat_h, lat_w)
        const_end = _align(self._const, 4096)
        self.const_size = const_end - self.const_base
        # ---- real walk
        self._const = fixed_end
        self.steps = []
        self._free = {}
        self._scratch_addr = {}
        self.gn_params, self.conv_chunks = {}, {}
        self._dry = False
        self._cur = self.arena_end = const_end
        self.macs = 0
        self.out_buf = self._walk(lat_h, lat_w)
        assert _align(self._const, 4096) == const_end
        limit = ue._tensor_dram_base - base
        assert self.arena_end - base <= limit, \
            f"VAE arena {(self.arena_end - base) / 1e6:.0f} MB exceeds the params arena {limit / 1e6:.0f} MB"
        return {"latent": [self.lat, lat_h, lat_w], "const_base": self.const_base, "const_size": self.const_size,
                "arena_end": self.arena_end, "input": self.in_buf.addr, "input_bytes": self.in_buf.lines * LINE,
                "output": self.out_buf["addr"], "output_bytes": self.out_buf["bytes"],
                "out_shape": self.out_buf["shape"], "pairs": [self.pairs_addr, self.pairs_bytes],
                "macs": self.macs,
                "steps": [{"kind": s["kind"], "name": s["name"], "out": s["out_shape"]} for s in self.steps]}

    # graph ------------------------------------------------------------
    def _walk(self, lat_h, lat_w):
        self.in_buf = self._new_buf("padded", self.lat, lat_h, lat_w)
        # the [x, 0] pair map SiLU needs: sized by the largest map, odd lines stay zero
        self.pairs_bytes = 2 * max(Buf.bytes_of("padded", c, h, w) for c, h, w in self._map_sizes(lat_h, lat_w))
        self.pairs_addr = self._alloc(self.pairs_bytes)
        self._step("zero_pairs", "pairs", None, {})
        x0 = self.conv(self.in_buf, "post_quant_conv", "padded")
        x = self.conv(x0, "decoder.conv_in", "padded")
        self._release(x0)
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
                u = self.upsample(x, f"decoder.up_blocks.{i}.upsamplers.0")
                self._release(x)
                x = self.conv(u, f"decoder.up_blocks.{i}.upsamplers.0.conv", "padded")
                self._release(u)
        g = self.gn(x, "decoder.conv_norm_out")
        self._release(x)
        h = self.silu(g, "decoder.conv_norm_out")
        self._release(g)
        out = self.conv(h, "decoder.conv_out", "tiled")
        self._release(h)
        return out

    def _map_sizes(self, lat_h, lat_w):
        c = self.block_out[-1]
        h, w = lat_h, lat_w
        yield (c, h, w)
        for i, cout in enumerate(reversed(self.block_out)):
            yield (cout, h, w)
            if i < len(self.block_out) - 1:
                h, w = 2 * h, 2 * w
                yield (cout, h, w)

    def resnet(self, x, p, cin, cout):
        g1 = self.gn(x, p + ".norm1")
        s1 = self.silu(g1, p + ".norm1")
        self._release(g1)
        h = self.conv(s1, p + ".conv1", "padded")
        self._release(s1)
        g2 = self.gn(h, p + ".norm2")
        s2 = self.silu(g2, p + ".norm2")
        self._release(g2)
        h2 = self.conv(s2, p + ".conv2", "padded")
        self._release(s2)
        self._release(h)
        if cin != cout:
            xs = self.conv(x, p + ".conv_shortcut", "padded")
            self._release(x)
            x = xs
        out = self.add(x, h2, p)
        self._release(x)
        self._release(h2)
        return out

    def attention(self, x, p):
        h = self.gn(x, p + ".group_norm")
        q = self.conv(h, p + ".to_q", "compact")
        k = self.conv(h, p + ".to_k", "compact")
        v = self.conv(h, p + ".to_v", "compact")
        self._release(h)
        o = self.attention_core(q, k, v, p)
        for b in (q, k, v):
            self._release(b)
        o2 = self.conv(o, p + ".to_out.0", "padded")
        self._release(o)
        out = self.add(x, o2, p)
        self._release(x)
        self._release(o2)
        return out

    def _step(self, kind, name, out, plan):
        shape = list(out.shape()) if isinstance(out, Buf) else out
        st = {"kind": kind, "name": name, "out_shape": shape, **plan}
        self.steps.append(st)
        return st

    # ------------------------------------------------------------------ op plans
    def conv(self, x: Buf, key: str, out_layout: str):
        OC, wc_c, kh, kw, pad = self.shapes[key]
        C, H, Wd = x.shape()
        assert wc_c == C, f"{key}: weight C_in {wc_c} != activation C {C}"
        if kh == kw == 1 and C % UE_VECTOR_SIZE == 0 and OC % UE_VECTOR_SIZE == 0 and out_layout != "tiled":
            return self.conv1x1(x, key, out_layout)
        OC_eff = OC if (OC % UE_VECTOR_SIZE == 0 or out_layout == "tiled") else _align(OC, UE_VECTOR_SIZE)
        g = self._conv_geometry(C, H, Wd, OC_eff, kh, kw, pad)
        n_oc, nck = g["oc_chunk"], g["n_chunks"]
        block_bytes = n_oc * g["blocks_per_oc"] * UE_VECTOR_SIZE
        assert block_bytes <= udc.URAM_NEAR_FULL_SIZE
        if not self._dry and key in self.conv_chunks:
            raise RuntimeError(key)
        WBLK = [self._calloc(block_bytes) for _ in range(nck)]
        SA = [self._calloc(n_oc * g["blocks_per_oc"] * 2) for _ in range(nck)]
        BA = [self._calloc(n_oc * 2) for _ in range(nck)]
        self.conv_chunks[key] = dict(WBLK=WBLK, SA=SA, BA=BA, block_bytes=block_bytes, oc_chunk=n_oc,
                                     gather=g["use_gather"], OC_eff=OC_eff, OC=OC)
        wexp = self._scratch("wexp", nck * _align(block_bytes * g["npix"], 4096))
        WEXP = [wexp + ci * _align(block_bytes * g["npix"], 4096) for ci in range(nck)]
        if out_layout == "tiled":
            out_bytes = nck * g["chunk_out"]
            out = {"addr": self._alloc(out_bytes), "bytes": out_bytes, "shape": [OC, g["out_h"], g["out_w"]],
                   "tiles": g["tiles"], "oc_chunk": n_oc}
            out_buf = None
        else:
            out_buf = self._new_buf(out_layout, OC, g["out_h"], g["out_w"])      # ct(OC) == ct(OC_eff)
            out = out_buf
        # work split: every (chunk, group) tile range split evenly across engines
        per_engine = [[] for _ in range(self.ne)]
        for ci in range(nck):
            for gi, grp in enumerate(g["groups"]):
                for e, (a, b) in enumerate(_split(grp[1] - grp[0], self.ne)):
                    if b > a:
                        per_engine[e].append((ci, gi, grp[0] + a, grp[0] + b))
        macs = g["out_h"] * g["out_w"] * OC * kh * kw * C
        self.macs += macs
        self._step("conv", key, out_buf if out_buf is not None else out["shape"],
                   dict(x=x, out=out, g=g, key=key, macs=macs, kh=kh, kw=kw, pad=pad, C=C, OC=OC_eff, n_oc=n_oc, nck=nck,
                        WEXP=WEXP, WBLK=WBLK, SA=SA, BA=BA, block_bytes=block_bytes, work=per_engine,
                        out_layout=out_layout))
        return out if out_buf is None else out_buf

    def conv1x1(self, x: Buf, key: str, out_layout: str):
        """[pixels, C] @ W[OC, C]^T + bias, one matmul per image row (M = W)."""
        OC, C, _, _, _ = self.shapes[key]
        _, H, Wd = x.shape()
        codes = self._calloc(OC * C)
        scales = self._calloc(OC * (C // UE_VECTOR_SIZE) * 2)
        bias = self._calloc(OC * 2)
        self.conv_chunks[key] = dict(mm=True, codes=codes, scales=scales, bias=bias, OC=OC, C=C)
        out = self._new_buf(out_layout, OC, H, Wd)
        self.macs += H * Wd * OC * C
        self._step("conv1x1", key, out, dict(x=x, out=out, C=C, OC=OC, W=Wd, codes=codes, scales=scales, bias=bias,
                                             ranges=_split(H, self.ne), macs=H * Wd * OC * C))
        return out

    def gn(self, x: Buf, key: str):
        C, H, Wd = x.shape()
        ct = x.ct
        assert x.layout == "padded"
        (_, slots, slot_lines, n_full, tail, flush_every, _nf) = udc.plan_group_norm(C, x.Hp, x.Wp, self.groups)
        n_chunks = n_full + (1 if tail else 0)
        chunk_bytes = slot_lines * LINE
        assert 2 * slot_lines <= 4096 and 4 * slot_lines <= 4096
        ranges = _split(n_chunks, self.ne)
        n_fl = [-(-(b - a) // flush_every) if b > a else 0 for a, b in ranges]
        F = sum(n_fl)
        bpp = max(1, K_PIECE // slots)                             # flush blocks per reduction piece
        pieces = [(f0, min(F, f0 + bpp)) for f0 in range(0, F, bpp)]
        gamma = self._calloc(ct * LINE)
        beta = self._calloc(ct * LINE)
        self.gn_params[key] = (gamma, beta)
        SUM = self._scratch("gn_SUM", F * chunk_bytes)             # all engines' flush blocks, contiguous
        SQ = self._scratch("gn_SQ", F * chunk_bytes)
        sum_off = [sum(n_fl[:e]) * chunk_bytes for e in range(self.ne)]
        # per-engine scratch for the redundant reduction / fold
        per = [dict(T=self._scratch(f"gn_T{e}", C * bpp * slots * 2), S=self._scratch(f"gn_S{e}", 64 * C * 2),
                    P=self._scratch(f"gn_P{e}", 64 * C * 2), ROW=self._scratch(f"gn_ROW{e}", 8 * ct * LINE),
                    REP=self._scratch(f"gn_REP{e}", 3 * chunk_bytes)) for e in range(self.ne)]
        out = self._new_buf("padded", C, H, Wd)
        self._step("gn", key, out, dict(x=x, out=out, C=C, ct=ct, slots=slots, slot_lines=slot_lines, n_full=n_full,
                                        tail=tail, flush_every=flush_every, ranges=ranges, n_fl=n_fl, F=F,
                                        pieces=pieces, SUM=SUM, SQ=SQ, sum_off=sum_off, per=per, gamma=gamma,
                                        beta=beta, chunk_bytes=chunk_bytes, n_pix=H * Wd))
        return out

    def silu(self, x: Buf, key: str):
        C, H, Wd = x.shape()
        total_lines = x.lines
        round_lines = min(total_lines, 0xFFF // 4, (udc.URAM_NEAR_FULL_SIZE // LINE) // 2)
        n_rounds = -(-total_lines // round_lines)
        chunk_bytes = round_lines * LINE
        assert n_rounds * chunk_bytes * 2 <= self.pairs_bytes
        out = self._new_buf("padded", C, H, Wd)
        ranges = _split(n_rounds, self.ne)
        SCR = [[self._scratch(f"silu_{e}_{j}", chunk_bytes) for j in range(3)] if b > a else None
               for e, (a, b) in enumerate(ranges)]
        self._step("silu", key + ".silu", out, dict(x=x, out=out, total_lines=total_lines, round_lines=round_lines,
                                          n_rounds=n_rounds, chunk_bytes=chunk_bytes, ranges=ranges, SCR=SCR))
        return out

    def add(self, x: Buf, y: Buf, key: str):
        assert x.shape() == y.shape() and x.layout == y.layout == "padded"
        total_lines = x.lines
        round_lines = min(udc.URAM_NEAR_FULL_SIZE // LINE, 4096)
        n_rounds = -(-total_lines // round_lines)
        out = self._new_buf("padded", *x.shape())
        self._step("add", key, out, dict(x=x, y=y, out=out, total_lines=total_lines, round_lines=round_lines,
                                         ranges=_split(n_rounds, self.ne)))
        return out

    def upsample(self, x: Buf, key: str):
        C, H, Wd = x.shape()
        out = self._new_buf("padded", C, 2 * H, 2 * Wd)
        self._step("upsample", key, out, dict(x=x, out=out, ranges=_split(H, self.ne)))
        return out

    def attention_core(self, q: Buf, k: Buf, v: Buf, key: str):
        C, H, Wd = q.shape()
        seq = H * Wd
        assert q.layout == "compact" and C % UE_VECTOR_SIZE == 0 and seq % UE_VECTOR_SIZE == 0
        out = self._new_buf("compact", C, H, Wd)
        ranges = _split(seq, self.ne, unit=UE_VECTOR_SIZE) if seq % (self.ne * UE_VECTOR_SIZE) == 0 \
            else [(0, seq)] + [(0, 0)] * (self.ne - 1)
        SCRATCH = [self._scratch(f"att_{e}", (C * seq + seq * seq + seq * C) * 2) if b > a else 0
                   for e, (a, b) in enumerate(ranges)]
        self._step("attention", key + ".attention", out, dict(q=q, k=k, v=v, out=out, seq=seq, C=C, ranges=ranges, SCRATCH=SCRATCH))
        return out

    # ------------------------------------------------------------------ emission
    def emit(self) -> None:
        """Emit the current engine's share of every step (called once per engine)."""
        self.eng = self.ue.cur
        self.e = self.ue.e
        self.eng._isa_reg_counter = max(self.eng._isa_reg_counter, getattr(self.ue, "FIRST_DYNAMIC_REG", 15))
        self.eng.reset_inst_ptr_counter()
        for st in self.steps[:getattr(self, "emit_steps", None)]:
            getattr(self, "_emit_" + st["kind"])(st)

    def _B(self):
        self.ue.B()

    # helpers ----------------------------------------------------------
    def _load(self, dram, sram, nbytes, chunk=0, jump=0):
        if chunk and chunk != jump:
            self.eng.accelerator_memory_to_sram(accelerator_dram_address=dram, sram_address=sram, element_size=0,
                                                memcpy_length_bytes=nbytes, stride_bytes_per_chunk=chunk,
                                                stride_jump_bytes=jump)
        else:
            self.eng.accelerator_memory_to_sram(accelerator_dram_address=dram, sram_address=sram, element_size=0,
                                                memcpy_length_bytes=nbytes)

    def _store(self, sram, dram, nbytes, chunk=0, jump=0):
        if chunk and chunk != jump:
            self.eng.sram_to_accelerator_memory(sram_address=sram, accelerator_dram_address=dram, element_size=0,
                                                memcpy_length_bytes=nbytes, stride_bytes_per_chunk=chunk,
                                                stride_jump_bytes=jump)
        else:
            self.eng.sram_to_accelerator_memory(sram_address=sram, accelerator_dram_address=dram, element_size=0,
                                                memcpy_length_bytes=nbytes)

    def _zero_borders(self, b: Buf):
        """Zero the one-pixel border of a padded map (zeros staged in URAM_B)."""
        if b.layout != "padded":
            return
        px = b.ct * LINE
        row = b.Wp * px
        assert row <= udc.URAM_NEAR_FULL_SIZE
        self._load(self.consts["ZERO"], URAM_B, max(row, b.H * px))
        self._store(URAM_B, b.addr, row)                                         # top row
        self._store(URAM_B, b.addr + (b.Hp - 1) * row, row)                      # bottom row
        self._store(URAM_B, b.addr + row, b.H * px, chunk=px, jump=row)          # left column
        self._store(URAM_B, b.addr + row + (b.Wp - 1) * px, b.H * px, chunk=px, jump=row)   # right column

    def _loop_store(self, sram, dram, nbytes, count):
        """Write the same URAM block ``count`` times to consecutive DRAM slots."""
        eng = self.eng
        reg = eng.alloc_isa_reg()
        eng.generate_instruction_add_set(reg, dram >> 3)
        start = eng.get_program_dram_addr()
        eng.generate_instruction_jump_abs(udc.ue_35bit_addr_shifter(
            start + (eng.capture_count + 1) * INSTRUCTION_SIZE_BYTES))
        eng.loop_start(loop_cnt=count)
        eng.sram_to_accelerator_memory(sram_address=sram, accelerator_dram_address=0, element_size=0,
                                       memcpy_length_bytes=nbytes, general_reg_src=reg)
        eng.generate_instruction_add_imm(src_reg_idx=reg, immediate_value=nbytes >> 3)
        eng.loop_end()
        eng.release_isa_reg()

    def _mm(self, M, K, N, A, Bm, OUT):
        eng = self.eng
        reg = eng.alloc_isa_reg()
        eng.generate_instruction_add_set(reg, M)
        eng.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=A, B_DRAM_ADDR=Bm, OUTPUT_DRAM_ADDR=OUT, gpr_M_reg=reg)
        eng.release_isa_reg()
        if self.e == 0:
            self.ue._flops_acc += 2 * M * K * N

    # zero the pair map once --------------------------------------------
    def _emit_zero_pairs(self, st):
        if self.e == 0:
            n = udc.URAM_NEAR_FULL_SIZE
            self._load(self.consts["ZERO"], 0, n)
            full, rem = divmod(self.pairs_bytes, n)
            if full:
                self._loop_store(0, self.pairs_addr, n, full)
            if rem:
                self._store(0, self.pairs_addr + full * n, rem)
        self._B()

    # conv -------------------------------------------------------------
    def _emit_conv(self, st):
        eng, e = self.eng, self.e
        g, x, out = st["g"], st["x"], st["out"]
        nck, n_oc, C = st["nck"], st["n_oc"], st["C"]
        kh, kw, pad = st["kh"], st["kw"], st["pad"]
        bb = st["block_bytes"]
        # 1. expand this engine's weight chunks (block -> npix copies)
        for ci in range(nck):
            if ci % self.ne == e:
                self._load(st["WBLK"][ci], 0, bb)
                self._loop_store(0, st["WEXP"][ci], bb, g["npix"])
        self._B()
        # 2. tiles
        lalu_mode, lalu_a, lalu_b = udc._conv_fused_lalu(False, False, False)
        ct_in = x.ct
        bsrc = x.border - pad                       # window coords are in the conv's own padding
        src_row = x.Wp * ct_in * LINE
        tiled = st["out_layout"] == "tiled"
        if not tiled:
            ct_out = out.ct
            dst_px = ct_out * LINE
            dst_row = out.Wp * dst_px
            OC = st["OC"]
        last_ci = None
        for ci, gi, t0, t1 in st["work"][e]:
            (start, stop, th, tw, win_h, win_w, _ao, _ab, out_off, out_b, _rl) = g["groups"][gi]
            if ci != last_ci:
                eng.accelerator_memory_to_scale_sram(st["SA"][ci], n_oc * g["blocks_per_oc"])
                eng.accelerator_memory_to_bias_sram(st["BA"][ci], n_oc)
                last_ci = ci
            for t in range(t0, t1):
                oy0, ox0, _th, _tw, y0, x0, _wh, _ww = g["tiles"][t]
                chunk = win_w * ct_in * LINE
                self._load(x.addr + ((y0 + bsrc) * x.Wp + x0 + bsrc) * ct_in * LINE, 0, win_h * chunk,
                           chunk=chunk, jump=src_row)
                eng.start_queue_for_conv2d_operation(
                    act_sram_start_addr=0, output_sram_wb_addr=WB, weights_dram_addr=st["WEXP"][ci],
                    kernel_w=kw, kernel_h=kh, ct=ct_in, oc_count=n_oc, out_w=tw, out_h=th, w_pad=win_w,
                    stride_s=1, data_type=TYPE.IF8, bias_enable=True, lalu_mode=lalu_mode, lalu_a=lalu_a,
                    lalu_b=lalu_b, dilation=1, gather=g["use_gather"], c_in=C)
                if tiled:
                    self._store(WB, out["addr"] + ci * g["chunk_out"] + out_off + (t - start) * out_b, out_b)
                elif n_oc == OC:
                    self._store(WB, out.addr + ((oy0 + out.border) * out.Wp + ox0 + out.border) * dst_px,
                                th * tw * dst_px, chunk=tw * dst_px, jump=dst_row)
                else:
                    for py in range(th):
                        self._store(WB + py * tw * n_oc * 2,
                                    out.addr + ((oy0 + py + out.border) * out.Wp + ox0 + out.border) * dst_px + ci * n_oc * 2,
                                    tw * n_oc * 2, chunk=n_oc * 2, jump=dst_px)
        if not tiled and e == 0:
            self._zero_borders(out)
        if e == 0:
            self.ue._flops_acc += 2 * st["macs"]
        self._B()

    def _emit_conv1x1(self, st):
        eng, e = self.eng, self.e
        x, out, C, OC, Wd = st["x"], st["out"], st["C"], st["OC"], st["W"]
        r0, r1 = st["ranges"][e]
        if r1 > r0:
            reg = eng.alloc_isa_reg()
            eng.generate_instruction_add_set(reg, Wd)
            for r in range(r0, r1):
                a = x.addr + ((r + x.border) * x.Wp + x.border) * C * 2
                o = out.addr + ((r + out.border) * out.Wp + out.border) * OC * 2
                eng.matmat_mul_core(M=Wd, K=C, N=OC, A_DRAM_ADDR=a, B_DRAM_ADDR=st["codes"], OUTPUT_DRAM_ADDR=o,
                                    is_B_quantized=True, data_type=TYPE.IF8, SCALE_DRAM_ADDR=st["scales"],
                                    C_DRAM_ADDR=st["bias"], bias_mode="broadcast_N", gpr_M_reg=reg)
            eng.release_isa_reg()
        if e == 0:
            self._zero_borders(out)
            self.ue._flops_acc += 2 * st["macs"]
        self._B()

    # GroupNorm --------------------------------------------------------
    def _emit_gn(self, st):
        eng, e = self.eng, self.e
        x, out, C, ct = st["x"], st["out"], st["C"], st["ct"]
        slots, slot_lines, n_full, tail, fe = st["slots"], st["slot_lines"], st["n_full"], st["tail"], st["flush_every"]
        cb = st["chunk_bytes"]
        c0, c1 = st["ranges"][e]
        per = st["per"][e]
        ZERO = self.consts["ZERO"]
        a_ln = lambda ln: ln * LINE                 # noqa: E731
        b_ln = lambda ln: URAM_B + ln * LINE        # noqa: E731
        A_CHUNK, A_SQ = 0, slot_lines
        B_REP, B_ZERO, B_COPY, B_ACC = 0, slot_lines, 2 * slot_lines, 3 * slot_lines
        n_elem = ct * UE_VECTOR_SIZE

        def take_lines(i):
            return (slots if i < n_full else tail) * ct

        def pass_(dst_region, centred):
            """Accumulate x (or (x - mean)^2) lane-wise over this engine's chunks into flush blocks."""
            if c1 <= c0:
                return
            base = dst_region + st["sum_off"][e]
            self._load(ZERO, b_ln(B_ZERO), cb)
            if centred:
                self._load(per["REP"], b_ln(B_REP), cb)             # mean repeated per slot
            self._load(ZERO, b_ln(B_ACC), cb)
            for i in range(c0, c1):
                tl = take_lines(i)
                tb = tl * LINE
                self._load(x.addr + i * cb, a_ln(A_CHUNK), tb)
                if centred:
                    eng.eltwise_sub_core(a_ln(A_CHUNK), b_ln(B_REP), a_ln(A_CHUNK), tl * UE_VECTOR_SIZE)
                    eng.eltwise_add_core(a_ln(A_CHUNK), b_ln(B_ZERO), b_ln(B_COPY), tl * UE_VECTOR_SIZE)
                    eng.eltwise_mul_core(a_ln(A_CHUNK), b_ln(B_COPY), a_ln(A_CHUNK), tl * UE_VECTOR_SIZE)
                eng.eltwise_add_core(a_ln(A_CHUNK), b_ln(B_ACC), b_ln(B_ACC), tl * UE_VECTOR_SIZE)
                j = i - c0
                if (j + 1) % fe == 0 or i == c1 - 1:
                    self._store(b_ln(B_ACC), base + (j // fe) * cb, cb)
                    if i != c1 - 1:
                        self._load(ZERO, b_ln(B_ACC), cb)

        def reduce_(region, scale):
            """Column sums of all flush blocks (ladder, in pieces of <= K_PIECE rows) -> group-averaged
            row (scaled) in URAM_A line 0."""
            # the running sum lives in DRAM between pieces: the matmul core stages through URAM
            acc = per["ROW"] + 4 * ct * LINE
            for pi, (f0, f1) in enumerate(st["pieces"]):
                K = (f1 - f0) * slots
                self._mm(C, C, K, self.consts[f"IDENT{C}"], region + f0 * cb, per["T"])    # T = blocks^T [C, K]
                self._mm(64, K, C, self.consts["ONES"], per["T"], per["S"])                # S[0] = column sums
                self._load(per["S"], a_ln(0), ct * LINE)
                if pi > 0:
                    self._load(acc, b_ln(0), ct * LINE)
                    eng.eltwise_add_core(a_ln(0), b_ln(0), a_ln(0), n_elem)
                self._store(a_ln(0), acc, ct * LINE)
            self._store(a_ln(0), per["S"], ct * LINE)                                      # S[0] = total sums
            self._mm(64, C, C, per["S"], self.consts[f"GAVG{C}"], per["P"])                # P[0] = group averages
            self._load(per["P"], a_ln(0), ct * LINE)
            eng.broadcast_mul(scale, a_ln(0), a_ln(0), n_elem)

        def rep_(src_line, dst):
            """Repeat the ct-line row at URAM_A ``src_line`` ``slots`` times into DRAM ``dst``."""
            self._loop_store(a_ln(src_line), dst, ct * LINE, slots)

        # ---- pass 1: sums
        pass_(st["SUM"], centred=False)
        self._B()
        if c1 <= c0:                       # nothing to normalise on this engine
            self._B()
            self._B()
            return
        reduce_(st["SUM"], 1.0 / st["n_pix"])                                       # A0 = mean_c
        self._store(a_ln(0), per["ROW"], ct * LINE)                                 # ROW[0] = mean
        rep_(0, per["REP"])
        # ---- pass 2: centred squares
        pass_(st["SQ"], centred=True)
        self._B()
        reduce_(st["SQ"], 1.0 / st["n_pix"])                                        # A0 = var_g
        eng.broadcast_add(self.gn_eps, a_ln(0), a_ln(0), n_elem)                    # A0 = v = var + eps
        # Newton rsqrt: y <- y * (1.5 - 0.5 * v * y^2), from y0 = 2^-10
        self._load(ZERO, b_ln(0), ct * LINE)                                         # B0 = zeros
        eng.eltwise_add_core(a_ln(0), b_ln(0), b_ln(ct), n_elem)                      # B1 = v
        self._load(self.consts["Y0"], a_ln(ct), ct * LINE)                           # A1 = y
        for _ in range(NEWTON_ITERS):
            eng.eltwise_add_core(a_ln(ct), b_ln(0), b_ln(2 * ct), n_elem)             # B2 = y
            eng.eltwise_mul_core(a_ln(ct), b_ln(2 * ct), a_ln(2 * ct), n_elem)        # A2 = y^2
            eng.eltwise_mul_core(a_ln(2 * ct), b_ln(ct), a_ln(2 * ct), n_elem)        # A2 = v y^2
            eng.broadcast_mul(-0.5, a_ln(2 * ct), a_ln(2 * ct), n_elem)
            eng.broadcast_add(1.5, a_ln(2 * ct), a_ln(2 * ct), n_elem)                # A2 = 1.5 - 0.5 v y^2
            eng.eltwise_add_core(a_ln(2 * ct), b_ln(0), b_ln(3 * ct), n_elem)         # B3 = that
            eng.eltwise_mul_core(a_ln(ct), b_ln(3 * ct), a_ln(ct), n_elem)            # A1 = y * (...)
        # A = gamma * inv_std ; B = beta - mean * A
        self._load(st["gamma"], b_ln(2 * ct), ct * LINE)                             # B2 = gamma
        eng.eltwise_mul_core(a_ln(ct), b_ln(2 * ct), a_ln(3 * ct), n_elem)           # A3 = A_c
        self._load(per["ROW"], b_ln(3 * ct), ct * LINE)                              # B3 = mean
        eng.eltwise_mul_core(a_ln(3 * ct), b_ln(3 * ct), a_ln(4 * ct), n_elem)       # A4 = mean * A
        eng.eltwise_add_core(a_ln(4 * ct), b_ln(0), b_ln(4 * ct), n_elem)            # B4 = mean * A
        self._load(st["beta"], a_ln(5 * ct), ct * LINE)                              # A5 = beta
        eng.eltwise_sub_core(a_ln(5 * ct), b_ln(4 * ct), a_ln(5 * ct), n_elem)       # A5 = B_c
        rep_(3 * ct, per["REP"])                                                     # REP = A repeated
        rep_(5 * ct, per["REP"] + cb)                                                # REP + cb = B repeated
        # ---- pass 3: apply
        self._load(per["REP"], b_ln(0), 2 * cb)
        for i in range(c0, c1):
            tl = take_lines(i)
            tb = tl * LINE
            self._load(x.addr + i * cb, a_ln(A_CHUNK), tb)
            eng.eltwise_mul_core(a_ln(A_CHUNK), b_ln(0), a_ln(A_CHUNK), tl * UE_VECTOR_SIZE)
            eng.eltwise_add_core(a_ln(A_CHUNK), b_ln(slot_lines), a_ln(A_CHUNK), tl * UE_VECTOR_SIZE)
            self._store(a_ln(A_CHUNK), out.addr + i * cb, tb)
        if e == 0:
            self._zero_borders(out)
        self._B()

    # SiLU -------------------------------------------------------------
    def _emit_silu(self, st):
        eng, e = self.eng, self.e
        x, out = st["x"], st["out"]
        rl, cb, n_rounds = st["round_lines"], st["chunk_bytes"], st["n_rounds"]
        r0, r1 = st["ranges"][e]
        if r1 > r0:
            # 1. build [x, 0] pairs for this engine's rounds (odd lines of the pair map are zero)
            for r in range(r0, r1):
                take = min(rl, st["total_lines"] - r * rl)
                self._load(x.addr + r * cb, 0, take * LINE)
                self._store(0, self.pairs_addr + r * 2 * cb, take * LINE, chunk=LINE, jump=2 * LINE)
            # 2. the layer driver's SiLU loop over the pair rounds
            XP, XN, WQ = st["SCR"][e]
            coeffs = udc.SILU_RECIP_POLY
            pair_bytes = 2 * cb
            elements = rl * UE_VECTOR_SIZE
            pair_reg = eng.alloc_isa_reg()
            out_reg = eng.alloc_isa_reg()
            eng.generate_instruction_add_set(pair_reg, (self.pairs_addr + r0 * pair_bytes) >> 3)
            eng.generate_instruction_add_set(out_reg, (out.addr + r0 * cb) >> 3)
            start = eng.get_program_dram_addr()
            eng.generate_instruction_jump_abs(udc.ue_35bit_addr_shifter(
                start + (eng.capture_count + 1) * INSTRUCTION_SIZE_BYTES))
            eng.loop_start(loop_cnt=r1 - r0)
            eng.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0, element_size=0,
                                           memcpy_length_bytes=pair_bytes, general_reg_src=pair_reg)
            eng.start_queue_for_maxpool2d_operation(act_sram_start_addr=0, output_sram_wb_addr=URAM_B,
                                                    kernel_w=2, kernel_h=1, out_w=rl, out_h=1,
                                                    w_pad=2 * rl, stride_s=2)
            self._store(URAM_B, XP, cb)
            eng.accelerator_memory_to_sram(accelerator_dram_address=0, sram_address=0, element_size=0,
                                           memcpy_length_bytes=cb, stride_bytes_per_chunk=LINE,
                                           stride_jump_bytes=2 * LINE, general_reg_src=pair_reg)
            eng.eltwise_sub_core(0, URAM_B, 0, elements)
            self._store(0, XN, cb)
            eng.eltwise_sub_core(0, URAM_B, 0, elements)
            eng.exp_core(0, 0, elements)
            self._store(0, WQ, cb)
            self._load(WQ, URAM_B, cb)
            eng.broadcast_mul(float(coeffs[-1]), 0, 0, elements)
            eng.broadcast_add(float(coeffs[-2]), 0, 0, elements)
            for coeff in reversed(coeffs[:-2]):
                eng.eltwise_mul_core(0, URAM_B, 0, elements)
                eng.broadcast_add(float(coeff), 0, 0, elements)
            eng.eltwise_mul_core(0, URAM_B, URAM_B, elements)
            self._store(URAM_B, WQ, cb)
            self._load(XP, URAM_B, cb)
            eng.eltwise_mul_core(0, URAM_B, 0, elements)
            self._store(0, XP, cb)
            self._load(XN, 0, cb)
            self._load(WQ, URAM_B, cb)
            eng.eltwise_mul_core(0, URAM_B, 0, elements)
            self._load(XP, URAM_B, cb)
            eng.eltwise_add_core(0, URAM_B, 0, elements)
            eng.sram_to_accelerator_memory(sram_address=0, accelerator_dram_address=0, element_size=0,
                                           memcpy_length_bytes=cb, general_reg_src=out_reg)
            eng.generate_instruction_add_imm(src_reg_idx=pair_reg, immediate_value=pair_bytes >> 3)
            eng.generate_instruction_add_imm(src_reg_idx=out_reg, immediate_value=cb >> 3)
            body = eng.loop_end()
            assert body <= 256
            eng.release_isa_reg()
            eng.release_isa_reg()
        self._B()

    # residual add ------------------------------------------------------
    def _emit_add(self, st):
        eng, e = self.eng, self.e
        x, y, out, rl, total = st["x"], st["y"], st["out"], st["round_lines"], st["total_lines"]
        r0, r1 = st["ranges"][e]
        for r in range(r0, r1):
            done = r * rl
            take = min(rl, total - done)
            off, nb = done * LINE, take * LINE
            self._load(x.addr + off, 0, nb)
            self._load(y.addr + off, URAM_B, nb)
            eng.eltwise_add_core(0, URAM_B, 0, take * UE_VECTOR_SIZE)
            self._store(0, out.addr + off, nb)
        self._B()

    # nearest 2x upsample -----------------------------------------------
    def _emit_upsample(self, st):
        e = self.e
        x, out = st["x"], st["out"]
        px = x.ct * LINE
        row_in = x.Wp * px
        row_out = out.Wp * px
        y0, y1 = st["ranges"][e]
        assert x.W * px <= udc.URAM_NEAR_FULL_SIZE
        for y in range(y0, y1):
            self._load(x.addr + ((y + 1) * x.Wp + 1) * px, 0, x.W * px)
            for a in (0, 1):
                for b in (0, 1):
                    self._store(0, out.addr + ((2 * y + a + 1) * out.Wp + 1 + b) * px, x.W * px,
                                chunk=px, jump=2 * px)
        if e == 0:
            self._zero_borders(out)
        self._B()

    # mid-block attention ------------------------------------------------
    def _emit_attention(self, st):
        eng, e = self.eng, self.e
        q, k, v, out, seq, C = st["q"], st["k"], st["v"], st["out"], st["seq"], st["C"]
        r0, r1 = st["ranges"][e]
        if r1 > r0:
            eng.unified_attention_core(batch=r1 - r0, aligned_seq_len=seq, head_dim=C,
                                       Q_DRAM_ADDR=q.addr + r0 * C * 2, K_DRAM_ADDR=k.addr, V_DRAM_ADDR=v.addr,
                                       BIAS_DRAM_ADDR=self.consts["ATT_BIAS"] + r0 * seq * 2,
                                       OUTPUT_DRAM_ADDR=out.addr + r0 * C * 2, SCRATCH_DRAM_ADDR=st["SCRATCH"][e],
                                       IDENTITY_DRAM_ADDR=self.consts["IDENT64"])
            if e == 0:
                self.ue._flops_acc += 4 * seq * seq * C
        self._B()

    # ------------------------------------------------------------------ host side
    def const_image(self, weights: dict) -> bytes:
        """The constant image (weights blocks / scales / biases, GN affine rows, matrices)."""
        img = bytearray(self.const_size)
        base = self.const_base

        def put(addr, data: bytes):
            assert addr - base + len(data) <= len(img)
            img[addr - base:addr - base + len(data)] = data
        put(self.consts["ONES"], _i16bytes(torch.ones(64 * 8192, dtype=torch.bfloat16)))
        put(self.consts["Y0"], _i16bytes(torch.full((512,), 2.0 ** -10, dtype=torch.bfloat16)))
        cpg = None
        for name, addr in self.consts.items():
            if name.startswith("IDENT"):
                c = int(name[5:])
                put(addr, _i16bytes(torch.eye(c, dtype=torch.bfloat16)))
            elif name.startswith("GAVG"):
                c = int(name[4:])
                cpg = c // self.groups
                gavg = torch.zeros(c, c, dtype=torch.bfloat16)
                for g in range(self.groups):
                    gavg[g * cpg:(g + 1) * cpg, g * cpg:(g + 1) * cpg] = 1.0 / cpg
                put(addr, _i16bytes(gavg))
        for key, (gamma, beta) in self.gn_params.items():
            n = weights["norms"][key]
            C = n["gamma"].numel()
            ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
            for addr, vec in ((gamma, n["gamma"]), (beta, n["beta"])):
                line = torch.zeros(ct * UE_VECTOR_SIZE, dtype=torch.bfloat16)
                line[:C] = vec.to(torch.bfloat16)
                put(addr, _i16bytes(line))
        for key, cc in self.conv_chunks.items():
            c = weights["convs"][key]
            codes, bs, bias = c["codes"], c["block_scales"].to(torch.bfloat16), c["bias"]
            OC, C, kh, kw = codes.shape
            if cc.get("mm"):
                # quantized-B matmul operands: codes [OC, C] int8 row-major, scales [OC, C/64] bf16, bias [OC]
                assert (kh, kw) == (1, 1) and tuple(bs.shape) == (OC, C // UE_VECTOR_SIZE), key
                put(cc["codes"], (codes.reshape(OC, C).to(torch.int16) & 0xFF).to(torch.uint8).numpy().tobytes())
                put(cc["scales"], _i16bytes(bs))
                put(cc["bias"], _i16bytes(bias.to(torch.bfloat16)))
                continue
            if cc["OC_eff"] != OC:                      # pad the output channels with zero rows
                codes = torch.cat([codes, torch.zeros(cc["OC_eff"] - OC, C, kh, kw, dtype=codes.dtype)], 0)
                bs = torch.cat([bs, torch.ones(cc["OC_eff"] - OC, bs.shape[1], dtype=bs.dtype)], 0)
                bias = torch.cat([bias, torch.zeros(cc["OC_eff"] - OC, dtype=bias.dtype)], 0)
            n_oc, gather = cc["oc_chunk"], cc["gather"]
            ct = (C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
            taps, chunks = kh * kw * ct, (kh * kw * C + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE
            for ci, oc0 in enumerate(range(0, cc["OC_eff"], n_oc)):
                wc = codes[oc0:oc0 + n_oc]
                sc = udc._conv2d_chunk_scale(bs, 1.0, oc0=oc0, n_oc=n_oc)
                if gather:
                    blk = udc.conv2d_pack_weight_stream_gather(wc, 1, 1, TYPE.IF8)
                    ss = udc.conv2d_pack_scale_stream_gather(sc, n_oc, chunks)
                else:
                    blk = udc.conv2d_pack_weight_stream(wc, 1, 1, TYPE.IF8)
                    ss = udc.conv2d_pack_scale_stream(sc, n_oc, taps, 1, 1)
                assert blk.numel() == cc["block_bytes"], key
                put(cc["WBLK"][ci], blk.numpy().tobytes())
                put(cc["SA"][ci], _i16bytes(ss))
                put(cc["BA"][ci], _i16bytes(udc.conv2d_pack_bias_stream(bias[oc0:oc0 + n_oc], 1, 1)))
        return bytes(img)

    def pack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """(32, h, w) -> the padded channels-in-lanes map the first conv reads."""
        return udc.conv2d_pack_activation_map(latent.to(torch.bfloat16), 1)

    def unpack_output(self, flat: torch.Tensor) -> torch.Tensor:
        o = self.out_buf
        return udc.conv2d_unpack_tiled_result(flat, o["tiles"], o["shape"][1], o["shape"][2], o["shape"][0], o["oc_chunk"])

    def read_buf(self, ue, b: Buf) -> torch.Tensor:
        """Host readback of a padded / compact map as (C, H, W) (debug / probe)."""
        flat = ue.dma_from_accelerator_memory(b.addr, (b.lines * UE_VECTOR_SIZE,))
        full = udc.packed_map_to_chw(flat, b.Hp, b.Wp, b.C)
        if b.layout == "padded":
            return full[:, 1:-1, 1:-1].contiguous()
        return full


# ----------------------------------------------------------------------------
# Host oracle with the same IF8 weights
# ----------------------------------------------------------------------------

def reference_decode(weights: dict, cfg: dict, latent: torch.Tensor, dtype=torch.float32,
                     device: str = "cpu", intermediates: dict | None = None) -> torch.Tensor:
    """Same graph on the host from the dequantized IF8 codes (bias in bf16).
    ``intermediates`` (a dict) receives every op's output keyed like the on-device steps."""
    v = cfg["vae"]
    groups, eps = v["norm_num_groups"], v.get("gn_eps", 1e-6)
    convs, norms = weights["convs"], weights["norms"]
    x = latent[None].to(device=device, dtype=dtype)

    def rec(key, t):
        if intermediates is not None:
            intermediates[key] = t[0].detach().cpu()
        return t

    def conv(x, key):
        c = convs[key]
        w = W.dequantize_conv_if8(c["codes"], c["block_scales"]).to(device=device, dtype=dtype)
        return rec(key, F.conv2d(x, w, c["bias"].to(device=device, dtype=dtype), padding=c["pad"]))

    def gn(x, key):
        n = norms[key]
        return rec(key, F.group_norm(x, groups, n["gamma"].to(device=device, dtype=dtype),
                                     n["beta"].to(device=device, dtype=dtype), eps))

    def silu(x, key):
        return rec(key + ".silu", F.silu(x))

    def resnet(x, p, cin, cout):
        h = conv(silu(gn(x, p + ".norm1"), p + ".norm1"), p + ".conv1")
        h = conv(silu(gn(h, p + ".norm2"), p + ".norm2"), p + ".conv2")
        if cin != cout:
            x = conv(x, p + ".conv_shortcut")
        return rec(p, x + h)

    def attn(x, p):
        B, C, H, Wd = x.shape
        h = gn(x, p + ".group_norm")
        q, k, vv = (conv(h, p + f".{n}").reshape(B, C, H * Wd).transpose(1, 2) for n in ("to_q", "to_k", "to_v"))
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], vv[:, None])[:, 0]
        rec(p + ".attention", o.transpose(1, 2).reshape(B, C, H, Wd))
        o = conv(o.transpose(1, 2).reshape(B, C, H, Wd), p + ".to_out.0")
        return rec(p, x + o)

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
            x = rec(f"decoder.up_blocks.{i}.upsamplers.0", F.interpolate(x, scale_factor=2.0, mode="nearest"))
            x = conv(x, f"decoder.up_blocks.{i}.upsamplers.0.conv")
    x = silu(gn(x, "decoder.conv_norm_out"), "decoder.conv_norm_out")
    return conv(x, "decoder.conv_out")[0].cpu()


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
