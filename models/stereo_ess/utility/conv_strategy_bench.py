#!/usr/bin/env python3
"""Board benchmark of every conv the repo's models run: native CONV2D walker vs im2col + matmul vs Winograd.

    python models/stereo_ess/utility/conv_strategy_bench.py --list        # host only: the unique shapes, no board
    python models/stereo_ess/utility/conv_strategy_bench.py               # board, 8 engines, resumes a partial run
    python models/stereo_ess/utility/conv_strategy_bench.py --report      # host only: graphs + CONV_STRATEGY.md

Input: the conv inventory written by the code screeners, scratchpad/conv_strategy/inventory/*.jsonl (one record per
model layer: shape, input size, executions per inference, weight format, where it runs). Records collapse to unique
bench shapes; every shape runs the three lowerings on the same bf16 input with bf16 weights (IF4 / IF8 layers are
benchmarked in bf16 too), relu(conv(x) + b), checked against torch fp32:

  native16   CONV2D opcode, FLOAT16 weights with weight_reuse (gather mode when Cin is small enough), each engine
             on its own band of output rows (columns for 1D / short maps), window staged from a halo map.
  direct     im2col by strided DMA + one bf16 matmul per row chunk, K = kh*kw*Cin (a 1x1 / stride 1 is the matmul
             on the map itself).
  winograd   F(2x2,3x3) (2D) or F(2,3) (1D, kernel 1 x k): transforms as SRAM add/subs, 16 (4) K=Cin matmuls per
             tile; dilation through d x d parity sub-grids, kernels above 3 through 3-tap pieces accumulated in the M
             planes. n/a for stride > 1, 1x1, and kernels where the pieces save no multiplies (2, 4, ...).

Shapes the three lowerings do not cover are listed, not run: depthwise / grouped convs (no matmul shape), layers the
pipeline does not execute. Transposed convs run as their sub-pixel equivalent (stride-1 conv with ceil(k/s) taps and
Cout*s outputs, then depth-to-space, which is how bigcodec / unet lower them). Channel counts that are not multiples
of 64 are zero-padded to the 64-lane vector (as the board does); GFLOPS use the real shape.

Results: scratchpad/conv_strategy/results_e<N>.json, rewritten after every shape (a rerun skips finished shapes).
--report: scratchpad/conv_strategy/img/<type>_perf.png + <type>_snr.png and scratchpad/conv_strategy/CONV_STRATEGY.md.
"""
import contextlib
import glob
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from conv3x3_paths_probe import (ACT_URAM_LINES, IM2COL_BUDGET, SRAM_BANK, URAM_B_BYTES, WB_SRAM,  # noqa: E402
                                 Probe, get, put, scopy, split, up128)

OUT_DIR = os.path.join(ROOT, "scratchpad", "conv_strategy")
INV_GLOB = os.path.join(OUT_DIR, "inventory", "*.jsonl")
PATHS = ("native16", "direct", "winograd")
IM2COL_VARIANT = "auto"                   # --im2col-variant: auto | pertap (one tap per copy) | lanes64 (no compaction)
LIB = False                               # --lib: run the library router (conv_strategies via conv2d_prepare/auto)
LIB_SCRATCH = 64 << 20
LIB_DTYPE = None                          # --lib-dtype if8 | if4: router quantizes; SNR vs the dequantized weights
NATIVE_PLANES = False                     # --native-planes: chunked walker output to per-chunk planes (no scatter)
WINO_BUDGET = 128 << 20                   # bytes of Winograd D + M planes per engine; bigger bands go in row groups
SHORT = {"flux2_klein_4b": "flux2", "qwen2.5_omni_7b": "qwen-omni", "qwen2.5_vl_3b": "qwen2.5-vl",
         "qwen3.5_2b": "qwen3.5", "locateanything_3b": "locateanything", "dpdfnet8khz": "dpdfnet8k",
         "gemma4_e2b": "gemma4-e2b", "gemma4_e4b": "gemma4-e4b", "mobilenetv2": "mobilenetv2"}


class HMap:
    """HWC map with zero halos, separate for rows (hh) and columns (hw): a 1D signal needs no row halo."""

    def __init__(self, H, W, C, hh, hw, addr=0):
        self.H, self.W, self.C, self.hh, self.hw, self.addr = H, W, C, hh, hw, addr
        self.pb = C * 2
        self.Wp, self.Hp = W + 2 * hw, H + 2 * hh
        self.pitch = self.Wp * self.pb
        self.npix = self.Hp * self.Wp + 2 * self.Wp + 320

    def pix(self, y, x):
        return self.addr + ((y + self.hh) * self.Wp + x + self.hw) * self.pb


class NA(Exception):
    """This lowering does not apply to the shape (reported as n/a with the reason)."""


def up64(n):
    return (n + 63) // 64 * 64


# ----------------------------------------------------------------------------------------------- inventory
def load_inventory(pattern=INV_GLOB):
    recs = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            recs.extend(json.loads(ln) for ln in fh if ln.strip())
    return recs


def _side_pad(p, k, d):
    """Per-side pad. The screeners give the TOTAL for asymmetric (causal / TF-same) padding: a pad wider than a
    'same' conv's is a total, benchmarked as the symmetric split (same work, output within one pixel)."""
    p = p or 0
    return -(-p // 2) if p > d * (k - 1) // 2 else p


def bench_shape(r):
    """Inventory record -> (bench shape dict, type label, note) or (None, reason, None)."""
    op = r.get("op")
    if op in (None, "none"):
        return None, "no conv", None
    if (r.get("notes") or "").startswith("NOT_EXECUTED"):
        return None, "not executed by the pipeline", None
    cin, cout, g = r["cin"], r["cout"], r.get("groups") or 1
    kh, kw, sh, sw = r["kh"], r["kw"], r["sh"] or 1, r["sw"] or 1
    dh, dw = r["dh"] or 1, r["dw"] or 1
    H, W = r["in_h"], r["in_w"]
    if g != 1:
        return None, ("depthwise" if g == cin else f"grouped ({g} groups)"), None
    note = ""
    if op == "conv3d":                                   # video patch embed on one image: temporal taps fold into Cin
        kt = 2
        note = f"conv3d patch embed, temporal {kt} folded into Cin"
        cin *= kt
    if op.startswith("conv_transpose"):
        label = "ConvTranspose (as sub-pixel conv)"
        note = f"ConvTranspose k{kh}x{kw} s{sh}x{sw} {cin}->{cout}: {sh * sw} phases of a {-(-kh // sh)}x{-(-kw // sw)} conv"
        kh, kw, cout = -(-kh // sh), -(-kw // sw), cout * sh * sw
        sh = sw = dh = dw = 1
        ph, pw = kh // 2, kw // 2
    else:
        ph, pw = _side_pad(r["ph"], kh, dh), _side_pad(r["pw"], kw, dw)
        if H == 1 and kh == 1:
            label = f"Conv1d k={kw}" if sw == 1 else "Conv1d strided (downsampling)"
            sh, dh, ph = sw, dw, 0
        elif kh == kw == sh == sw and kh > 1 and not (r["ph"] or r["pw"]):
            label = "Conv2d patchify (kernel = stride)"
        elif min(kh, kw) >= 5 and sh == sw > 1:
            label = "Conv2d large-kernel stem (strided)"
        else:
            label = (f"Conv2d {kh}x{kw}" + (f" stride {sh}" if sh == sw and sh > 1 else "")
                     + (f" stride {sh}x{sw}" if sh != sw else "") + (f" dilation {dh}" if dh > 1 else ""))
    shape = dict(H=H, W=W, Cin=cin, Cout=cout, kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw)
    return shape, label, note


def parse_conv(text):
    """'KHxKW sS pP dD | HxW | Cin->Cout' (stride / pad / dilation optional, pad may be PHxPW) -> shape dict."""
    import re
    m = re.fullmatch(r"\s*(\d+)x(\d+)((?:\s+[spd][\dx]+)*)\s*\|\s*(\d+)x(\d+)\s*\|\s*(\d+)\s*->\s*(\d+)\s*",
                     text)
    if not m:
        raise SystemExit(f"--conv {text!r}: expected e.g. '3x3 s1 p1 d1 | 64x64 | 256->256'")
    kh, kw, opts, H, W, ci, co = m.groups()
    kh, kw, H, W = int(kh), int(kw), int(H), int(W)
    o = {t[0]: t[1:] for t in opts.split()}
    two = lambda v, dflt: tuple(int(x) for x in v.split("x")) if v and "x" in v else (int(v or dflt),) * 2
    sh, sw = two(o.get("s"), 1)
    dh, dw = two(o.get("d"), 1)
    ph, pw = two(o.get("p"), None) if "p" in o else ((dh * (kh - 1)) // 2, (dw * (kw - 1)) // 2)
    if H == 1 and kh == 1:
        sh, dh, ph = sw, dw, 0
    return dict(H=H, W=W, Cin=int(ci), Cout=int(co), kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw)


def shape_id(s):
    return (f"{s['kh']}x{s['kw']} s{s['sh']}x{s['sw']} p{s['ph']}x{s['pw']} d{s['dh']}x{s['dw']} "
            f"| {s['H']}x{s['W']} | {s['Cin']}->{s['Cout']}")


def collect(recs):
    """Unique bench shapes {sid: {"shape", "label", "users": [record...], "notes"}} and the records not benched."""
    shapes, skipped = {}, []
    for r in recs:
        shape, label, note = bench_shape(r)
        if shape is None:
            if r.get("op") not in (None, "none"):
                skipped.append((r, label))
            continue
        sid = shape_id(shape)
        ent = shapes.setdefault(sid, {"shape": shape, "label": label, "users": [], "notes": set()})
        ent["users"].append(r)
        if note:
            ent["notes"].add(note)
    return shapes, skipped


# ----------------------------------------------------------------------------------------------- one shape
class BSpec:
    def __init__(self, sid, shape, label):
        self.sid, self.label = sid, label
        for k_, v in shape.items():
            setattr(self, k_, v)
        self.Ci, self.Co = up64(self.Cin), up64(self.Cout)
        self.Hout = (self.H + 2 * self.ph - self.dh * (self.kh - 1) - 1) // self.sh + 1
        self.Wout = (self.W + 2 * self.pw - self.dw * (self.kw - 1) - 1) // self.sw + 1
        self.wino, self.wino_why = winograd_plan(self)

    @property
    def nominal(self):
        return 2.0 * self.Hout * self.Wout * self.kh * self.kw * self.Cin * self.Cout


def winograd_plan(sp):
    """How F(2x2,3x3) / F(2,3) reaches this conv, or (None, why not)."""
    if sp.kh == 1 and sp.kw == 1:
        return None, "1x1: no input is shared between outputs (it is a plain matmul)"
    if sp.sh != 1 or sp.sw != 1:
        return None, f"stride {sp.sh if sp.sh == sp.sw else f'{sp.sh}x{sp.sw}'}: no F(2,3) piece in its polyphase split"
    if sp.kh == 1:
        n = -(-sp.kw // 3)
        if 2 * n >= sp.kw:
            return None, f"1x{sp.kw}: 3-tap pieces save no multiplies ({2 * n} vs {sp.kw} per output)"
        if sp.ph:
            return None, "row padding on a 1-row kernel"
        return dict(dim=1, gs=sp.dw, nph=1, npw=n, mults=(2 * n, sp.kw)), None
    if sp.kw == 1:
        return None, f"{sp.kh}x1 column kernel (not wired)"
    if sp.dh != sp.dw:
        return None, "unequal dilations"
    nph, npw = -(-sp.kh // 3), -(-sp.kw // 3)
    if 4 * nph * npw >= sp.kh * sp.kw:
        return None, f"{sp.kh}x{sp.kw}: 3x3 pieces save no multiplies ({4 * nph * npw} vs {sp.kh * sp.kw} per output)"
    return dict(dim=2, gs=sp.dh, nph=nph, npw=npw, mults=(4 * nph * npw, sp.kh * sp.kw)), None


def wino_desc(w):
    if w is None:
        return None
    base = "F(2x2,3x3)" if w["dim"] == 2 else "F(2,3)"
    extra = []
    if w["gs"] > 1:
        extra.append(f"{w['gs'] ** w['dim']} dilation sub-grids")
    if w["nph"] * w["npw"] > 1:
        extra.append(f"{w['nph'] * w['npw']} kernel pieces")
    return base + (" via " + " + ".join(extra) if extra else "")


def bands(nr, nc, NE):
    """Per-engine (row0, row1, col0, col1): output-row bands, or column bands when there are fewer rows than engines."""
    if nr >= NE or nc < NE:
        return [split(nr, i, NE) + (0, nc) for i in range(NE)], "rows"
    return [(0, nr) + split(nc, i, NE) for i in range(NE)], "cols"


def wino_chunk(TW, Cin, Cout):
    """Widest tile-row chunk whose staging fits one SRAM bank: 12 input-side and 18 output-side slots."""
    for twc in range(TW, 0, -1):
        if 18 * up128(twc * Cout * 2) <= SRAM_BANK and 12 * up128((twc + 1) * Cin * 2) <= SRAM_BANK:
            return twc
    return None


def bench_case(probe, sp, paths):
    import torch
    import torch.nn.functional as F
    import user_dma_core
    from user_dma_core import TYPE, calculate_snr
    ue, NE = probe.ue, probe.NE
    H, W, Cin, Cout, Ci, Co = sp.H, sp.W, sp.Cin, sp.Cout, sp.Ci, sp.Co
    kh, kw, sh, sw, ph, pw, dh, dw = sp.kh, sp.kw, sp.sh, sp.sw, sp.ph, sp.pw, sp.dh, sp.dw
    Hout, Wout = sp.Hout, sp.Wout
    nominal = sp.nominal
    torch.manual_seed(0)
    x = torch.relu(torch.randn(Cin, H, W))
    g = torch.randn(Cout, Cin, kh, kw) * (1.0 / (kh * kw * Cin) ** 0.5)
    b = torch.randn(Cout) * 0.1
    xq, gq, bq = (t.to(torch.bfloat16).float() for t in (x, g, b))
    ref = torch.relu(F.conv2d(xq[None], gq, bq, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw)))[0]
    dtq = None
    if LIB and LIB_DTYPE:                                 # reference on the weights the board actually uses
        import quant_lib
        import user_dma_core as U
        dtq = {"if8": U.TYPE.IF8, "if4": U.TYPE.IF4}[LIB_DTYPE]
        wd = torch.zeros(up64(Cout), kh, kw, up64(Cin))
        wd[:Cout, :, :, :Cin] = gq.permute(0, 2, 3, 1)
        data, sc = U._conv_quantize(wd.reshape(up64(Cout), -1), dtq)
        deq = (quant_lib.dequantize_if8 if LIB_DTYPE == "if8" else quant_lib.dequantize_if4)(
            data, sc, up64(Cout), kh * kw * up64(Cin), block_size=64)
        gdq = torch.as_tensor(deq).float().view(up64(Cout), kh, kw, up64(Cin)).permute(0, 3, 1, 2)[:Cout, :Cin]
        ref = torch.relu(F.conv2d(xq[None], gdq, bq, stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw)))[0]
    gp = torch.zeros(Co, Ci, kh, kw)
    gp[:Cout, :Cin] = gq
    bp = torch.zeros(Co)
    bp[:Cout] = bq
    reps = probe.args.reps or max(1, min(16, int(2e-3 / max(nominal / (0.5 * probe.peak * 1e9), 1e-6))))
    print(f"== {sp.label} | {sp.sid} -> {Hout}x{Wout}  {nominal / 1e9:.3f} GFLOP  reps {reps}", flush=True)
    xmaps = {}
    t_limit = getattr(ue, "_program_dram_base", None) or ue._tensor_dram_base + 0x2000_0000

    def tal(n):                                           # allocate_tensor_dram has no bound: never run into programs
        a = probe.al(n)
        if ue.get_tensor_dram_addr() > t_limit:
            raise MemoryError(f"maps need more than the {(t_limit - ue._tensor_dram_base) >> 20} MB tensor region")
        return a

    def xmap(hh, hw, C=Ci):
        if (hh, hw, C) not in xmaps:
            X = HMap(H, W, C, hh, hw)
            X.addr = tal(X.npix * X.pb)
            grid = torch.zeros(X.npix, C)
            grid[:X.Hp * X.Wp].view(X.Hp, X.Wp, C)[hh:hh + H, hw:hw + W, :Cin] = xq.permute(1, 2, 0)
            put(ue, X.addr, grid)
            xmaps[(hh, hw, C)] = X
        return xmaps[(hh, hw, C)]

    def ymap(hh, hw):
        m = HMap(Hout, Wout, Co, hh, hw)
        m.addr = tal(m.npix * m.pb)
        put(ue, m.addr, torch.zeros(m.npix * Co))
        return m

    one_d = H == 1 and kh == 1                            # 1D signal: no row halo
    hh0, hw0 = (0 if one_d else max(1, ph)), max(1, pw)

    # ---- direct: im2col + matmul
    def setup_direct():
        pointwise = kh == 1 and kw == 1 and sh == 1 and sw == 1
        # small Cin: compact 32-channel pixels (64 B, whole AXI beats) and K zero-padded to the 64-lane vector,
        # instead of 64-lane pixels that would inflate K by 64 / Cin (a 3-channel 16x16 patchify: 8192, not 16384)
        Cd = 32 if Cin <= 32 and not pointwise and IM2COL_VARIANT != "lanes64" else Ci
        Xd = xmap(hh0, hw0, Cd)
        K = up64(kh * kw * Cd)
        if K > 16320:
            raise NA(f"K = {K} is above one matmul's 16320 (K/64 <= 255)")
        bb, axis = bands(Hout, Wout, NE)
        trick = sh == 1 and sw == 1 and Hout == H and Wout == W and axis == "rows"
        Yd = ymap(1, Xd.hw) if trick else ymap(0, 0)
        gd = torch.zeros(Co, kh, kw, Cd)
        gd[:Cout, :, :, :Cin] = gq.permute(0, 2, 3, 1)
        Wd = torch.zeros(Co, K)
        Wd[:, :kh * kw * Cd] = gd.reshape(Co, kh * kw * Cd)
        P = []
        for i in range(NE):
            ra, rb, ca, cb = bb[i]
            nc = cb - ca
            row_bytes = (Xd.Wp if trick else max(nc, 1)) * K * 2
            rc = max(1, min(rb - ra, IM2COL_BUDGET // row_bytes))
            if not trick and (pointwise or not (ca == 0 and cb == Wout)):
                rc = 1                                    # output rows not contiguous: one matmul per row
            st = {"bd": probe.pwt(i, bp, "bias"), "wd": probe.pwt(i, Wd, "w_direct"), "rc": rc, "band": bb[i]}
            if not pointwise:
                st["im"] = probe.pal(i, rc * row_bytes, "im2col", zero=K > kh * kw * Cd)   # zero K tail once
            P.append(st)
        return {"P": P, "X": Xd, "Y": Yd, "trick": trick, "K": K, "pointwise": pointwise, "C": Cd,
                # board: compact 32-ch rows of 448 B stepping by 128 B (7x7 s2, Cin 3) came back at 8 dB; per-tap
                # copies and 64-lane rows both pass (2026-11-06). auto avoids that pattern: wide compact strided rows.
                "pertap": IM2COL_VARIANT == "pertap" or (IM2COL_VARIANT == "auto" and Cd == 32 and sw > 1
                                                         and kw * Cd * 2 > 256 and (kw * Cd * 2) % 128)}

    def direct(e, i, S):
        st, Xd, Yd, K, Cd = S["P"][i], S["X"], S["Y"], S["K"], S["C"]
        ra, rb, ca, cb = st["band"]
        nc = cb - ca
        if rb <= ra or nc <= 0:
            return 0
        gM = e.alloc_isa_reg()
        for c0 in range(ra, rb, st["rc"]):
            n = min(st["rc"], rb - c0)
            if S["trick"]:                                # 'same' stride-1 conv: one pass over n padded rows
                Mr = n * Xd.Wp
                if S["pointwise"]:
                    a_addr = Xd.pix(c0, 0)
                else:
                    a_addr = st["im"]
                    for ky in range(kh):
                        if dw == 1 and not S["pertap"]:
                            scopy(e, Xd.pix(c0 + ky * dh - ph, -pw), st["im"] + ky * kw * Cd * 2, Mr, kw * Xd.pb,
                                  src_jump=Xd.pb, dst_jump=K * 2)
                        else:
                            for kx in range(kw):
                                scopy(e, Xd.pix(c0 + ky * dh - ph, kx * dw - pw), st["im"] + (ky * kw + kx) * Cd * 2,
                                      Mr, Xd.pb, dst_jump=K * 2)
                out = Yd.pix(c0, 0)
            else:                                         # per output row, exactly Wout (or the band's) pixels
                Mr = n * nc
                if S["pointwise"]:
                    a_addr = Xd.pix(c0 - ph, ca - pw)
                else:
                    a_addr = st["im"]
                    for r in range(n):
                        oy = c0 + r
                        dst = st["im"] + r * nc * K * 2
                        for ky in range(kh):
                            iy = oy * sh + ky * dh - ph
                            if dw == 1 and not S["pertap"]:
                                scopy(e, Xd.pix(iy, ca * sw - pw), dst + ky * kw * Cd * 2, nc, kw * Xd.pb,
                                      src_jump=sw * Xd.pb, dst_jump=K * 2)
                            else:
                                for kx in range(kw):
                                    scopy(e, Xd.pix(iy, ca * sw + kx * dw - pw), dst + (ky * kw + kx) * Cd * 2, nc,
                                          Xd.pb, src_jump=sw * Xd.pb, dst_jump=K * 2)
                out = Yd.pix(c0, ca)
            e.generate_instruction_add_set(gM, Mr)
            e.matmat_mul_core(M=Mr, K=K, N=Co, A_DRAM_ADDR=a_addr, B_DRAM_ADDR=st["wd"], OUTPUT_DRAM_ADDR=out,
                              C_DRAM_ADDR=st["bd"], bias_mode="broadcast_N", gpr_M_reg=gM, clamp_enable=True,
                              clamp_min=0.0, clamp_max=float("inf"))
        if S["trick"] and Yd.hw:                          # halo columns the padded rows wrote: re-zero (own rows)
            n = rb - ra
            chunk, jump = 2 * Yd.hw * Yd.pb, Yd.Wp * Yd.pb
            e.accelerator_memory_to_sram(probe.ZEROS, 0, n * chunk // 2)
            e.sram_to_accelerator_memory(0, Yd.pix(ra, Wout), n * chunk // 2, stride_bytes_per_chunk=chunk,
                                         stride_jump_bytes=jump)
        return 1

    # ---- native CONV2D walker, FLOAT16 weights, weight_reuse
    def setup_native():
        if kh > 15 or kw > 15:
            raise NA(f"kernel {kh}x{kw} exceeds the walker's 4-bit kernel fields (max 15)")
        one_row = H == 1 and kh == 1 and Hout == 1
        if not one_row and sh != sw:
            raise NA(f"stride {sh}x{sw}: the walker has one stride for both axes")
        if kh > 1 and kw > 1 and dh != dw:
            raise NA("unequal dilations")
        s = sw if one_row else sh
        d = dw if kh == 1 else dh
        can_gather = Cin <= 255 and kh * kw * Cin <= 256
        chunks = -(-kh * kw * Cin // 64)
        ct = Ci // 64
        taps = kh * kw * ct
        use_gather = can_gather and max(kh * kw * ((Cin + 3) // 4), Co * chunks) < Co * taps
        Xn = xmap(hh0, hw0)
        bb, axis = bands(Hout, Wout, NE)
        plans = {}
        for ra, rb, ca, cb in bb:
            n, nc = rb - ra, cb - ca
            if n <= 0 or nc <= 0 or (n, nc) in plans:
                continue
            row_stride = d * ((kw - 1) * d + 1) * ct                  # as plan_conv2d_layer_tiles checks it
            if row_stride > 0xFFF:
                raise NA(f"walker row-stride register: dilation {d} x window {(kw - 1) * d + 1} x ct "
                         f"{ct} = {row_stride} > 4095 (12-bit geometry field)")
            out_h, out_w, occ, tiles = user_dma_core.plan_conv2d_layer_tiles(
                c_in=Cin if use_gather else Ci, oc_count=Co, in_h=(n - 1) * s + (kh - 1) * d + 1,
                in_w=(nc - 1) * s + (kw - 1) * d + 1, kernel_h=kh, kernel_w=kw, stride_s=s, pad=0, pad_h=0,
                dilation=d, gather=use_gather, bias_enabled=True, weight_block_limit=4096, act_uram_addr=0,
                wb_uram_addr=ACT_URAM_LINES)
            assert (out_h, out_w) == (n, nc), (out_h, out_w, n, nc)
            plans[(n, nc)] = (occ, tiles, user_dma_core.conv2d_tile_geometry_groups(tiles))
        occ = next(iter(plans.values()))[0]
        assert all(v[0] == occ for v in plans.values()), plans
        beat = user_dma_core.ue_axi_beat_bytes_for(user_dma_core.UE_AXI_DATA_WIDTH_BITS or 512)
        if occ != Co and (occ * 2) % beat:
            raise NA(f"walker oc chunk {occ} ({occ * 2} B per pixel) is not whole {beat} B AXI beats: its writeback "
                     f"cannot be scattered (conv_walker_pattern_fits rejects it too)")
        blocks = chunks if use_gather else taps
        assert occ * blocks * 128 <= URAM_B_BYTES, ("pattern", occ, blocks)
        wsrc = gp
        if use_gather:
            wsrc = torch.zeros(Co, Cin, kh, kw)
            wsrc[:Cout] = gq
        wchunks = []
        for oc0 in range(0, Co, occ):
            sl = slice(oc0, oc0 + occ)
            if use_gather:
                wblk = user_dma_core.conv2d_pack_weight_stream_gather(wsrc[sl], 1, 1, TYPE.FLOAT16)
            else:
                wblk = user_dma_core.conv2d_pack_weight_stream(wsrc[sl], 1, 1, TYPE.FLOAT16)
            bblk = user_dma_core.conv2d_pack_bias_stream(bp[sl].to(torch.bfloat16), 1, 1)
            wchunks.append((oc0, wblk.numpy().tobytes(), user_dma_core.tensor_le_bytes(bblk)))
        print(f"   native16: {'gather' if use_gather else 'channels'} mode, oc chunk {occ} ({Co // occ} chunks), "
              f"pattern {occ * blocks * 128 // 1024} KB, {axis} split, tiles per band "
              f"{[len(v[1]) for v in plans.values()]}", flush=True)
        tile_px = max(int(t[2]) * int(t[3]) for v in plans.values() for t in v[1])
        Yn = ymap(0, 0)
        planes = None
        if NATIVE_PLANES and occ != Co:                   # one [Hout][Wout][occ] plane per oc chunk, like the driver
            pbytes = Hout * Wout * occ * 2
            base0 = tal(up128((Co // occ) * pbytes))
            planes = [HMap(Hout, Wout, occ, 0, 0, base0 + ci * pbytes) for ci in range(Co // occ)]
        P = [{"w": [(oc0, probe.pwb(i, wb, f"n16_w{oc0}"), probe.pwb(i, bb_, f"n16_b{oc0}"))
                    for oc0, wb, bb_ in wchunks],
              "wbs": probe.pal(i, tile_px * occ * 2, "n16_wb_bounce"), "band": bb[i]} for i in range(NE)]
        return {"P": P, "X": Xn, "Y": Yn, "planes": planes, "plans": plans, "occ": occ, "s": s, "d": d, "gather": use_gather,
                "ct": ct}

    def native(e, i, S):
        st, Xn, Yn, occ = S["P"][i], S["X"], S["Y"], S["occ"]
        ra, rb, ca, cb = st["band"]
        if rb <= ra or cb <= ca:
            return 0
        s, d = S["s"], S["d"]
        _, tiles, groups = S["plans"][(rb - ra, cb - ca)]
        mode, la, lb = probe.lalu_relu
        launches = 0
        for oc0, w_addr, b_addr in st["w"]:
            for start, stop, th, tw, win_h, win_w in groups:
                e.accelerator_memory_to_bias_sram(b_addr, occ)
                for tile in tiles[start:stop]:
                    oy, ox, th_, tw_, y0, x0, wh, ww = (int(v) for v in tile)
                    a = Xn.pix(ra * sh - ph + y0, ca * s - pw + x0)        # padded-window origin in the halo map
                    row = ww * Xn.pb
                    if row == Xn.pitch or wh == 1:                          # one contiguous run (1D maps: 1 row)
                        e.accelerator_memory_to_sram(a, 0, wh * row // 2)
                    else:
                        e.accelerator_memory_to_sram(a, 0, wh * row // 2, stride_bytes_per_chunk=row,
                                                     stride_jump_bytes=Xn.pitch)
                    e.start_queue_for_conv2d_operation(
                        act_sram_start_addr=0, output_sram_wb_addr=WB_SRAM, weights_dram_addr=w_addr,
                        kernel_w=kw, kernel_h=kh, ct=S["ct"], oc_count=occ, out_w=tw_, out_h=th_, w_pad=ww,
                        stride_s=s, data_type=TYPE.FLOAT16, bias_enable=True, lalu_mode=mode, lalu_a=la, lalu_b=lb,
                        dilation=d, gather=S["gather"], c_in=Cin if S["gather"] else Ci, weight_reuse=True)
                    launches += 1
                    oc_b = occ * 2
                    if S["planes"]:                                       # this chunk's own plane: contiguous rows
                        pl = S["planes"][oc0 // occ]
                        row = tw_ * oc_b
                        if th_ == 1 or row == pl.pitch:
                            e.sram_to_accelerator_memory(WB_SRAM, pl.pix(ra + oy, ca + ox), th_ * row // 2)
                        elif row % 128 == 0:
                            for r in range(th_):
                                e.sram_to_accelerator_memory(WB_SRAM + r * row, pl.pix(ra + oy + r, ca + ox), row // 2)
                        else:
                            bounce = st["wbs"]
                            e.sram_to_accelerator_memory(WB_SRAM, bounce, up128(th_ * row) // 2)
                            for r in range(th_):
                                e.accelerator_memory_to_sram(bounce + r * row, 0, row // 2)
                                e.sram_to_accelerator_memory(0, pl.pix(ra + oy + r, ca + ox), row // 2)
                    elif oc_b == Yn.pb:                                   # whole pixels per row
                        row = tw_ * Yn.pb
                        if th_ == 1 or row == Yn.pitch:
                            e.sram_to_accelerator_memory(WB_SRAM, Yn.pix(ra + oy, ca + ox), th_ * row // 2)
                        else:
                            e.sram_to_accelerator_memory(WB_SRAM, Yn.pix(ra + oy, ca + ox), th_ * row // 2,
                                                         stride_bytes_per_chunk=row, stride_jump_bytes=Yn.pitch)
                    elif th_ == 1 or (tw_ * oc_b) % 128 == 0:                 # oc chunk at its lane offset, per row
                        for r in range(th_):
                            e.sram_to_accelerator_memory(WB_SRAM + r * tw_ * oc_b,
                                                         Yn.pix(ra + oy + r, ca + ox) + oc0 * 2, tw_ * oc_b // 2,
                                                         stride_bytes_per_chunk=oc_b, stride_jump_bytes=Yn.pb)
                    else:                                                 # rows off the 128 B lines: bounce via DRAM
                        bounce = st["wbs"]
                        e.sram_to_accelerator_memory(WB_SRAM, bounce, up128(th_ * tw_ * oc_b) // 2)
                        for r in range(th_):
                            e.accelerator_memory_to_sram(bounce + r * tw_ * oc_b, 0, tw_ * oc_b // 2)
                            e.sram_to_accelerator_memory(0, Yn.pix(ra + oy + r, ca + ox) + oc0 * 2, tw_ * oc_b // 2,
                                                         stride_bytes_per_chunk=oc_b, stride_jump_bytes=Yn.pb)
        return launches

    # ---- Winograd: F(2x2,3x3) (2D) or F(2,3) (1D); d x d parity sub-grids for dilation, 3-tap kernel pieces
    G = torch.tensor([[1., 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])
    wp = sp.wino

    def wino_geom():
        gs, dim, nph, npw = wp["gs"], wp["dim"], wp["nph"], wp["npw"]
        TH = -(-(-(-Hout // gs)) // 2) if dim == 2 else Hout             # tile rows (2D) / output rows (1D)
        TW = -(-(-(-Wout // gs)) // 2)
        max_row = (gs * (2 * (TH - 1) + 3 + 3 * (nph - 1)) + gs - 1 - ph) if dim == 2 else Hout - 1 - ph
        max_col = gs * (2 * TW + 1 + 3 * (npw - 1)) + gs - 1 - pw
        hh = 0 if dim == 1 else max(1, ph, max_row - (H - 1))
        hw = max(1, pw, max_col - (W - 1))
        return TH, TW, hh, hw

    def setup_winograd():
        if wp is None:
            raise NA(sp.wino_why)
        gs, dim, nph, npw = wp["gs"], wp["dim"], wp["nph"], wp["npw"]
        TH, TW, hh, hw = wino_geom()
        Xw = xmap(hh, hw)
        Yw = ymap(2 * gs if dim == 2 else 0, 2 * gs)
        bb, axis = bands(TH, TW, NE)
        twc = wino_chunk(max(b_[3] - b_[2] for b_ in bb), Ci, Co)
        if twc is None:
            raise NA("one tile column does not fit the SRAM staging")
        gk = torch.zeros(Co, Ci, 3 * nph, 3 * npw)
        gk[:Cout, :Cin, :kh, :kw] = gq
        npl = 16 if dim == 2 else 4
        P = []
        for i in range(NE):
            ra, rb, ca, cb = bb[i]
            twe = cb - ca
            tg = max(1, min(max(rb - ra, 1), WINO_BUDGET // (npl * max(twe, 1) * (Ci + Co) * 2)))
            nt = tg * max(twe, 1)
            us = []
            for a in range(nph):
                for b_ in range(npw):
                    if dim == 2:
                        U = torch.einsum("ik,ockl,jl->ocij", G, gk[:, :, 3 * a:3 * a + 3, 3 * b_:3 * b_ + 3], G)
                        sg = lambda r, c: (-1.0 if r == 2 else 1.0) * (-1.0 if c == 2 else 1.0)
                        us.append([probe.pwt(i, (sg(r, c) * U[:, :, r, c]).contiguous(), f"us{a}{b_}_{r}{c}")
                                   for r in range(4) for c in range(4)])
                    else:
                        U = torch.einsum("ik,ock->oci", G, gk[:, :, 0, 3 * b_:3 * b_ + 3])
                        us.append([probe.pwt(i, ((-1.0 if c == 2 else 1.0) * U[:, :, c]).contiguous(), f"u{b_}_{c}")
                                   for c in range(4)])
            P.append({"band": bb[i], "tg": tg, "twc": twc, "bd": probe.pwt(i, bp, "bias"), "us": us,
                      "D": [probe.pal(i, nt * Ci * 2, f"D{p}") for p in range(npl)],
                      "M": [probe.pal(i, nt * Co * 2, f"M{p}") for p in range(npl)]})
        print(f"   winograd: {wino_desc(wp)}, {TH}x{TW} tiles per sub-grid, {axis} split, chunk {twc}, "
              f"multiplies per output {wp['mults'][0]} vs {wp['mults'][1]}", flush=True)
        return {"P": P, "X": Xw, "Y": Yw}

    def winograd(e, i, S):
        st, X, Yw = S["P"][i], S["X"], S["Y"]
        ra, rb, ca, cb = st["band"]
        if rb <= ra or cb <= ca:
            return 0
        gs, dim, nph, npw = wp["gs"], wp["dim"], wp["nph"], wp["npw"]
        twe, twc = cb - ca, st["twc"]
        gM = e.alloc_isa_reg()
        D, Mx = st["D"], st["M"]
        A0, B0 = 0x00000, SRAM_BANK
        add = lambda a, b_, o, n: e.eltwise_add_core(a, b_, o, n)
        sub = lambda a, b_, o, n: e.eltwise_sub_core(a, b_, o, n)
        sz = up128((twc + 1) * X.pb)
        Pout = up128(twc * Co * 2)
        chunks = [(c0, min(twc, cb - c0)) for c0 in range(ca, cb, twc)]
        pieces = [(a, b_) for a in range(nph) for b_ in range(npw)]
        subgrids = [(ga, gb) for ga in (range(gs) if dim == 2 else (0,)) for gb in range(gs)]
        npl = 16 if dim == 2 else 4
        col0 = lambda c0, par, b_, gb: gs * (2 * c0 + par + 3 * b_) + gb - pw        # even/odd array start column
        for ga, gb in subgrids:
            for g0 in range(ra, rb, st["tg"]):
                g1 = min(g0 + st["tg"], rb)
                nt = (g1 - g0) * twe
                for pi, (a, b_) in enumerate(pieces):
                    for ty in range(g0, g1):
                        for c0, n in chunks:
                            off = (ty - g0) * twe + (c0 - ca)
                            ne, nel = (n + 1) * Ci, n * Ci
                            if dim == 2:
                                DS = lambda r, par: (A0 + (r * 2 + par) * sz) if r < 2 else (B0 + ((r - 2) * 2 + par) * sz)
                                TA = lambda k_, par: A0 + (4 + k_ * 2 + par) * sz
                                TB = lambda k_, par: B0 + (4 + k_ * 2 + par) * sz
                                for r in range(4):                          # each source row once: even / odd pixels
                                    iy = gs * (2 * ty + r + 3 * a) + ga - ph
                                    for par in (0, 1):
                                        e.accelerator_memory_to_sram(X.pix(iy, col0(c0, par, b_, gb)), DS(r, par), ne,
                                                                     stride_bytes_per_chunk=X.pb,
                                                                     stride_jump_bytes=2 * gs * X.pb)
                                for T in (TA, TB):                          # rows: D0-D2, D1+D2, D1-D2, D1-D3
                                    for par in (0, 1):
                                        sub(DS(0, par), DS(2, par), T(0, par), ne)
                                        add(DS(1, par), DS(2, par), T(1, par), ne)
                                        sub(DS(1, par), DS(2, par), T(2, par), ne)
                                        sub(DS(1, par), DS(3, par), T(3, par), ne)
                                rows_t = [(TA(k_, 0), TA(k_, 1), TB(k_, 0), TB(k_, 1), k_ * 4) for k_ in range(4)]
                                Vs = [A0 + q * sz for q in range(4)]
                            else:
                                iy = ty - ph
                                AE, AO, BE, BO = A0, A0 + sz, B0, B0 + sz
                                for base, par in ((AE, 0), (AO, 1), (BE, 0), (BO, 1)):
                                    e.accelerator_memory_to_sram(X.pix(iy, col0(c0, par, b_, gb)), base, ne,
                                                                 stride_bytes_per_chunk=X.pb,
                                                                 stride_jump_bytes=2 * gs * X.pb)
                                rows_t = [(AE, AO, BE, BO, 0)]
                                Vs = [A0 + (2 + q) * sz for q in range(4)]
                            for ae, ao, be, bo, p0 in rows_t:               # cols: d0-d2, d1+d2, d1-d2, d1-d3
                                sub(ae, be + X.pb, Vs[0], nel)
                                add(ao, be + X.pb, Vs[1], nel)
                                sub(ao, be + X.pb, Vs[2], nel)
                                sub(ao, bo + X.pb, Vs[3], nel)
                                for q in range(4):
                                    e.sram_to_accelerator_memory(Vs[q], D[p0 + q] + off * Ci * 2, nel)
                    bias_plane = 5 if dim == 2 else 1
                    for xi in range(npl):                                   # pieces after the first accumulate
                        e.generate_instruction_add_set(gM, nt)
                        if pi == 0:
                            kw_ = dict(C_DRAM_ADDR=st["bd"], bias_mode="broadcast_N") if xi == bias_plane else {}
                        else:
                            kw_ = dict(C_DRAM_ADDR=Mx[xi], bias_mode="full_matrix")
                        e.matmat_mul_core(M=nt, K=Ci, N=Co, A_DRAM_ADDR=D[xi], B_DRAM_ADDR=st["us"][pi][xi],
                                          OUTPUT_DRAM_ADDR=Mx[xi], gpr_M_reg=gM, **kw_)
                nout = 4 if dim == 2 else 2
                Y = Mx[:nout]
                for ty in range(g0, g1):
                    for c0, n in chunks:
                        off = (ty - g0) * twe + (c0 - ca)
                        nel = n * Co
                        if dim == 2:
                            Msr = lambda r, c: A0 + c * Pout if r == 1 else B0 + ({0: 0, 2: 1, 3: 2}[r] * 4 + c) * Pout
                            Ssr = lambda i_, c: (A0 + (4 + i_) * Pout if c == 1
                                                 else B0 + (12 + i_ * 3 + {0: 0, 2: 1, 3: 2}[c]) * Pout)
                            T0, T1 = A0 + 6 * Pout, A0 + 7 * Pout
                            Ysr = lambda k_: A0 + (8 + k_) * Pout
                            for r in range(4):
                                for c in range(4):
                                    e.accelerator_memory_to_sram(Mx[r * 4 + c] + off * Co * 2, Msr(r, c), nel)
                            for c in range(4):                              # S0 = (M1 + M2) + M0, S1 = (M1 - M2) - M3
                                add(Msr(1, c), Msr(2, c), T0, nel)
                                add(T0, Msr(0, c), Ssr(0, c), nel)
                                sub(Msr(1, c), Msr(2, c), T1, nel)
                                sub(T1, Msr(3, c), Ssr(1, c), nel)
                            for i_ in range(2):                             # y0 = (S1 + S2) + S0, y1 = (S1 - S2) - S3
                                add(Ssr(i_, 1), Ssr(i_, 2), T0, nel)
                                add(T0, Ssr(i_, 0), Ysr(2 * i_), nel)
                                sub(Ssr(i_, 1), Ssr(i_, 2), T1, nel)
                                sub(T1, Ssr(i_, 3), Ysr(2 * i_ + 1), nel)
                        else:
                            M1, M0, M2, M3 = A0, B0, B0 + Pout, B0 + 2 * Pout
                            T0, T1 = A0 + Pout, A0 + 2 * Pout
                            Ysr = lambda k_: A0 + (3 + k_) * Pout
                            for p_, dst in ((1, M1), (0, M0), (2, M2), (3, M3)):
                                e.accelerator_memory_to_sram(Mx[p_] + off * Co * 2, dst, nel)
                            add(M1, M2, T0, nel)                            # y0 = (M1 + M2) + M0
                            add(T0, M0, Ysr(0), nel)
                            sub(M1, M2, T1, nel)                            # y1 = (M1 - M2) - M3
                            sub(T1, M3, Ysr(1), nel)
                        for k_ in range(nout):
                            e.sram_to_accelerator_memory(Ysr(k_), Y[k_] + off * Co * 2, nel)
                ro = nt * Co // 64
                for yp in Y:                                                # ReLU (identity-matmul clamp)
                    e.generate_instruction_add_set(gM, ro)
                    e.activation_core(M=ro, N=64, A_DRAM_ADDR=yp, OUTPUT_DRAM_ADDR=yp,
                                      IDENTITY_DRAM_ADDR=probe.I64, activation="clamp", gpr_M_reg=gM)
                for k_ in range(nout):                                      # scatter the output planes
                    i_, j = (k_ // 2, k_ % 2) if dim == 2 else (0, k_)
                    for ty in range(g0, g1):
                        oy = gs * (2 * ty + i_) + ga if dim == 2 else ty
                        scopy(e, Y[k_] + (ty - g0) * twe * Co * 2, Yw.pix(oy, gs * (2 * ca + j) + gb), twe, Co * 2,
                              dst_jump=2 * gs * Yw.pb)
        return npl * len(pieces) * len(subgrids)

    setups = {"direct": setup_direct, "native16": setup_native, "winograd": setup_winograd}
    bodies = {"direct": direct, "native16": native, "winograd": winograd}
    if LIB:                                               # the library router, exactly as a model calls it
        import user_dma_core as U
        lname = {"native16": "native", "direct": "naive", "winograd": "wino"}
        one_d = H == 1 and kh == 1

        def setup_lib(meth):
            if sh != sw and not one_d:
                raise NA(f"stride {sh}x{sw}: conv2d_core / the router take one stride")
            geom = dict(c_in=Cin, in_h=H, in_w=W, kernel_h=kh, kernel_w=kw, stride_s=sw if one_d else sh, pad=pw,
                        pad_h=ph, dilation=dw if kh == 1 else dh, oc_count=Cout)
            g = U.ConvGeom(**geom)
            why = U.conv_strategy_unsupported(g, lname[meth], dtq)
            if why:
                raise NA(why)
            if lname[meth] == "wino":
                step = 2 * g.wino["gs"]
                th = -(-Hout // step) if g.wino["dim"] == 2 else Hout
                tb_, _ = bands(th, -(-Wout // step), NE)
                rs = step if g.wino["dim"] == 2 else 1
                pb = [(a_ * rs, min(b_ * rs, Hout), c_ * step, min(d_ * step, Wout)) for a_, b_, c_, d_ in tb_]
            else:
                pb, _ = bands(Hout, Wout, NE)
            return {"geom": geom, "s": lname[meth], "X": xmap(ph, pw), "Y": ymap(0, 0), "bands": pb, "cache": {},
                    "scr": [probe.pal(i, LIB_SCRATCH, "lib_scratch") for i in range(NE)]}

        def body_lib(e, i, S):
            ra, rb, ca, cb = S["bands"][i]
            if rb <= ra or cb <= ca:
                return 0
            e.conv2d_auto(strategy=S["s"], weight=gq, bias=bq, ACT_DRAM_ADDR=S["X"].addr,
                          OUTPUT_DRAM_ADDR=S["Y"].addr, relu_enable=True, band=(ra, rb, ca, cb),
                          scratch_addr=S["scr"][i], scratch_bytes=LIB_SCRATCH, params_ue=ue, cache=S["cache"],
                          data_type=dtq, **S["geom"])
            return 1

        setups = {m: (lambda m=m: setup_lib(m)) for m in setups}
        bodies = {m: body_lib for m in bodies}
    out = {}
    def run_path(meth):
        base = {"path": meth, "sid": sp.sid, "engines": NE, "reps": reps}
        probe.restore_arena()
        try:
            S = setups[meth]()
        except NA as ex:
            print(f"  {meth:9s} n/a: {ex}", flush=True)
            return dict(base, status="n/a", reason=str(ex))
        except (AssertionError, MemoryError, ValueError, KeyError) as ex:
            print(f"  {meth:9s} skip (setup): {ex!r}"[:240], flush=True)
            return dict(base, status="skip", reason=f"setup: {ex!r}"[:240])
        print(f"  {meth:9s} compiling + running ...", flush=True)
        chatter = io.StringIO()                           # the driver's per-capture prints: kept out of the log
        with contextlib.redirect_stdout(chatter):
            built = probe.run(lambda e, i: bodies[meth](e, i, S), reps=reps)
        if isinstance(built, Exception):
            print("\n".join(chatter.getvalue().splitlines()[-8:]))
            print(f"  {meth:9s} skip (compile): {built!r}"[:240], flush=True)
            return dict(base, status="skip", reason=f"compile: {built!r}"[:240])
        ms, ninst, nl = built
        ym = S["Y"]
        if S.get("planes"):                               # per-chunk planes -> HWC on the host (as the driver does)
            pl = S["planes"]
            hw = get(ue, pl[0].addr, len(pl) * Hout * Wout * pl[0].C).view(len(pl), Hout, Wout, pl[0].C)
            hw = hw.permute(1, 2, 0, 3).reshape(Hout, Wout, Co)[:, :, :Cout].permute(2, 0, 1)
        else:
            hw = get(ue, ym.addr, ym.Hp * ym.Wp * Co).view(ym.Hp, ym.Wp, Co)
            hw = hw[ym.hh:ym.hh + Hout, ym.hw:ym.hw + Wout, :Cout].permute(2, 0, 1)
        snr = float(calculate_snr(ref.flatten(), hw.flatten()))
        bad = int((~torch.isfinite(hw)).sum())
        good = snr >= 30 and bad == 0
        gf = nominal / (ms * 1e-3) / 1e9
        print(f"  {meth:9s} {ms:9.3f} ms/conv  {gf:7.1f} GFLOPS ({100 * gf / probe.peak:5.1f}% of peak)  "
              f"{nl} launches/eng  SNR {snr:6.2f} dB  non-finite {bad}  {'PASS' if good else 'FAIL'}", flush=True)
        return dict(base, status="ok" if good else "fail", ms=ms, gflops=gf, snr=snr, launches=nl, instr=ninst,
                    nonfinite=bad, wb="planes" if S.get("planes") else "scatter",
                    variant=IM2COL_VARIANT if meth == "direct" else None)

    for meth in paths:
        mark, keys, pmark = ue.get_tensor_dram_addr(), set(xmaps), ue._next_params_dram_addr
        try:
            out[meth] = run_path(meth)
        finally:                                          # release this path's maps (the shared region is 512 MB)
            ue._tensor_dram_addr = mark
            ue._next_params_dram_addr = pmark             # --lib uploads weights to params DRAM per path
            for k_ in set(xmaps) - keys:
                del xmaps[k_]
    ue.reset_tensor_dram_addr()
    probe.al(64 * 64 * 2 + 4000 * 128)                     # keep I64 / ZEROS at the same addresses
    return out


# ----------------------------------------------------------------------------------------------- listing
TYPE_ORDER = ("Conv2d 3x3", "Conv2d 3x3 dilation", "Conv2d 3x3 stride 2", "Conv2d large-kernel stem",
              "Conv2d 1x1", "Conv2d 1x1 stride 2", "Conv2d 5x5", "Conv2d 7x7", "Conv2d 1x3", "Conv2d 3x1",
              "Conv2d 5x1", "Conv2d patchify", "Conv1d k=1", "Conv1d k=3", "Conv1d k=5", "Conv1d k=7",
              "Conv1d k=11", "Conv1d strided", "Conv1d", "ConvTranspose")


def type_rank(label):
    for i, p in enumerate(TYPE_ORDER):
        if label == p or label.startswith(p + " ") or (p in ("Conv1d", "ConvTranspose", "Conv2d patchify",
                                                             "Conv2d large-kernel stem", "Conv1d strided")
                                                       and label.startswith(p)):
            return (i, label)
    return (len(TYPE_ORDER), label)


def users_text(users, limit=3):
    agg = {}
    for r in users:
        m = SHORT.get(r["model"], r["model"])
        agg[m] = agg.get(m, 0) + int(r.get("count") or 1)
    items = sorted(agg.items(), key=lambda t: -t[1])
    s = ", ".join(f"{m} ×{c}" for m, c in items[:limit])
    return s + (f" +{len(items) - limit}" if len(items) > limit else "")


def list_inventory(shapes, skipped, NE=8):
    by_type = {}
    for sid, ent in shapes.items():
        by_type.setdefault(ent["label"], []).append((sid, ent))
    total = 0
    for label in sorted(by_type, key=type_rank):
        rows = by_type[label]
        gfl = sum(BSpec(sid, e["shape"], label).nominal * sum(int(r.get("count") or 1) for r in e["users"])
                  for sid, e in rows) / 1e9
        print(f"\n{label}: {len(rows)} shapes, {gfl:.1f} GFLOP per inference over all users")
        for sid, ent in sorted(rows, key=lambda t: BSpec(t[0], t[1]["shape"], label).nominal):
            sp = BSpec(sid, ent["shape"], label)
            total += 1
            print(f"  {sid:58s} -> {sp.Hout}x{sp.Wout}  {sp.nominal / 1e9:8.3f} GFLOP  wino: "
                  f"{wino_desc(sp.wino) or 'n/a'}  | {users_text(ent['users'])}")
    print(f"\n{total} unique shapes to bench")
    why = {}
    for r, reason in skipped:
        why.setdefault(reason, []).append(r)
    print("not benched:")
    for reason, rs in why.items():
        print(f"  {reason}: {len(rs)} records, {sum(int(r.get('count') or 1) for r in rs)} executions "
              f"({users_text(rs, 6)})")


# ----------------------------------------------------------------------------------------------- main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--reps", type=int, default=0, help="convs per program; 0 = auto (more for small convs)")
    ap.add_argument("--inventory", default=INV_GLOB)
    ap.add_argument("--paths", default=",".join(PATHS))
    ap.add_argument("--only", default=None, help="substring filter on the type label or shape id; ';' separates several")
    ap.add_argument("--max-gflop", type=float, default=None, help="skip shapes above this many GFLOP")
    ap.add_argument("--fresh", action="store_true", help="ignore finished shapes in the results file")
    ap.add_argument("--retry-skips", action="store_true", help="rerun paths whose saved result is a build 'skip'")
    ap.add_argument("--retry-fails", action="store_true", help="rerun paths whose saved result failed the SNR check")
    ap.add_argument("--conv", default=None,
                    help='one ad-hoc conv instead of the inventory, e.g. "3x3 s1 p1 d1 | 64x64 | 256->256" '
                         '(1D: "1x7 s1 p3 d1 | 1x9000 | 384->384"); runs it every way and prints the fastest')
    ap.add_argument("--lib", action="store_true",
                    help="run the library router (ue.conv2d_auto(strategy=...)); own results file")
    ap.add_argument("--lib-dtype", default=None, choices=("if8", "if4"), help="--lib with quantized weights")
    ap.add_argument("--im2col-variant", default="auto", choices=("auto", "pertap", "lanes64"))
    ap.add_argument("--native-planes", action="store_true",
                    help="chunked native output goes to one plane per oc chunk (contiguous writes, host reorders)")
    ap.add_argument("--list", action="store_true", help="print the unique shapes and exit (host only)")
    ap.add_argument("--report", action="store_true", help="graphs + markdown from the results (host only)")
    args = ap.parse_args()
    global NATIVE_PLANES, IM2COL_VARIANT, LIB, LIB_DTYPE
    LIB = args.lib or bool(args.lib_dtype)
    LIB_DTYPE = args.lib_dtype
    NATIVE_PLANES = args.native_planes
    IM2COL_VARIANT = args.im2col_variant

    recs = load_inventory(args.inventory)
    shapes, skipped = collect(recs)
    if args.conv:
        shapes, skipped = {}, []
        shape = parse_conv(args.conv)
        lab = (f"Conv1d k={shape['kw']}" if shape["H"] == 1 and shape["kh"] == 1
               else f"Conv2d {shape['kh']}x{shape['kw']}")
        shapes[shape_id(shape)] = {"shape": shape, "label": lab, "notes": set(),
                                   "users": [{"model": "--conv", "count": 1, "op": "conv2d"}]}
    if args.list:
        list_inventory(shapes, skipped)
        return
    res_path = os.path.join(OUT_DIR, f"results_e{args.engines}{'_lib' if LIB else ''}{'_' + LIB_DTYPE if LIB_DTYPE else ''}"
                                     f"{'_adhoc' if args.conv else ''}.json")
    if args.report:
        import conv_strategy_report
        conv_strategy_report.report(shapes, skipped, res_path, OUT_DIR)
        conv_strategy_report.report_by_kernel(shapes, res_path, OUT_DIR)
        return

    saved = {}
    if os.path.exists(res_path):                          # always kept: --fresh only changes what gets rerun
        with open(res_path) as f:
            saved = json.load(f).get("results", {})
    paths = args.paths.split(",")
    todo = []
    for sid, ent in sorted(shapes.items(), key=lambda t: (type_rank(t[1]["label"]),
                                                         BSpec(t[0], t[1]["shape"], t[1]["label"]).nominal)):
        sp = BSpec(sid, ent["shape"], ent["label"])
        if args.only and not any(o.strip() in sp.label or o.strip() in sid for o in args.only.split(";")):
            continue
        if args.max_gflop and sp.nominal / 1e9 > args.max_gflop:
            continue
        done = {} if args.fresh else saved.get(sid, {})
        retry = {"skip"} if args.retry_skips else set()
        retry |= {"fail"} if args.retry_fails else set()
        if all(p in done and done[p]["status"] not in retry for p in paths):
            continue
        todo.append(sp)
    print(f"{len(shapes)} unique shapes, {len(todo)} to run ({len(shapes) - len(todo)} done or filtered)")
    probe = Probe(argparse.Namespace(dev=args.dev, engines=args.engines, reps=1))
    probe.args.reps = args.reps
    meta = {"engines": args.engines, "peak": probe.peak, "date": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.perf_counter()
    for n, sp in enumerate(todo):
        print(f"\n[{n + 1}/{len(todo)}] {time.perf_counter() - t0:7.0f}s", flush=True)
        try:
            saved.setdefault(sp.sid, {}).update(bench_case(probe, sp, paths))
        finally:
            with open(res_path + ".tmp", "w") as f:
                json.dump({"meta": meta, "results": saved}, f, indent=1)
            os.replace(res_path + ".tmp", res_path)
    print(f"\nresults: {res_path}  ({time.perf_counter() - t0:.0f}s)")
    if args.conv:
        for sp in todo:
            res = saved[sp.sid]
            ok = {p: res[p]["ms"] for p in paths if res.get(p, {}).get("status") == "ok"}
            name = {"native16": "native", "direct": "naive", "winograd": "wino"}
            print(f"\n{sp.sid}")
            for p in paths:
                r = res.get(p, {})
                txt = (f"{r['ms']:9.3f} ms  SNR {r['snr']:5.1f} dB" if r.get("status") in ("ok", "fail")
                       else f"{r.get('status')}: {r.get('reason', '')}")
                print(f"  {name[p]:7s} {txt}" + ("   <- fastest" if ok and p == min(ok, key=ok.get) else ""))
        return
    print("next: python models/stereo_ess/utility/conv_strategy_bench.py --report")


if __name__ == "__main__":
    main()
