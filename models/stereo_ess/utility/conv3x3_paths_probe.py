#!/usr/bin/env python3
"""Board probe: the four ways to run a 3x3 / stride 1 conv, same shapes, same 8-engine setup.

    python models/stereo_ess/utility/conv3x3_paths_probe.py                       # all cases, 8 engines
    python models/stereo_ess/utility/conv3x3_paths_probe.py --cases dec3 --engines 1
    python models/stereo_ess/utility/conv3x3_paths_probe.py --paths direct,native16

Question: for a Winograd-shaped 3x3 (stride 1, dilation 1, bf16), what is the ORDER of the paths now that the
CONV2D walker no longer re-streams its weights per output pixel (927fd8d RTL, UE_CONV_CTRL[30] weight_reuse)?
The answer sets user_dma_core.CONV3X3_BF16_FALLBACK, the bf16 lowering select_conv_path() uses when Winograd
does not apply. Each case also prints what select_conv_path() would pick for it today.

Paths (each engine works on its own row band, like the stereo --engines 8 program):
  direct     im2col in DRAM (3 overlapping row copies) + one runtime-M bf16 matmul, K = 9*Cin, bias + CLAMP ReLU.
  winograd   F(2x2,3x3) with both transforms in SRAM (the winograd_probe 'weo' flow): each source row loaded once
             as even / odd arrays, B^T d B as SRAM add/subs, 16 bf16 matmuls K = Cin, A^T M A in SRAM, ReLU, scatter.
             The model's fully fused flow (U resident in URAM B, matvec per tile) is faster still; this is the
             standalone upper bound on its cost.
  native16   hardware CONV2D opcode, FLOAT16 (bf16 storage) weights with weight_reuse: the conv's [oc][tap]
             pattern (128 B blocks) is loaded into URAM B once per launch and replayed for every tile pixel.
             Window staged from the halo map into URAM A, bias + CLAMP ReLU fused, writeback scattered to the map.
  native8    same launch with IF8 weights (64 B blocks, MixMSE INT8/FP8 per block + scale BRAM) -- the format
             flux2's VAE runs. Its SNR is reported against the bf16 reference AND against the dequantised weights.

Every path computes relu(conv(x) + b) on the same bf16 input and is checked against torch fp32 on the bf16 data.
Reported per path: ms per conv (median of 3 runs / reps), effective GFLOPS on the nominal 2*H*W*9*Cin*Cout
(Winograd executes 2.25x fewer MACs, so it can read above the matmul peak), SNR, launches per engine.

Cases (the real light_ess shapes, as winograd_probe): dec3 96x160 192->64 | enc1 48x80 64->64 |
dec1 24x40 640->256 | dec0 12x20 768->512.  Native oc chunking: 4096 blocks for FLOAT16, 8192 for IF8, and the
pattern must fit the 512 KiB URAM B (dec0 FLOAT16: 32 oc x 108 taps x 128 B = 442 KB per launch).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

import quant_lib
import user_dma_core
from user_dma_core import (CONV_GEOMETRY_QUEUE_CONFIG, DMA_DEVICE_C2H, DMA_DEVICE_H2C, TYPE, UE_MODE,
                           UnifiedEngine, calculate_snr, select_conv_path, set_dma_device)
from multi_engine_shard import MultiEngineScheduler, PrivateArena, model_multicore_layout

STAGE = 4000 * 128
V = 64
ACT_URAM_LINES = 0x300                 # padded window in URAM A lines [0, 0x300), writeback from 0x300
WB_SRAM = ACT_URAM_LINES << 7
URAM_B_BYTES = 512 * 1024
CASES = {"dec3": (96, 160, 192, 64, 2), "enc1": (48, 80, 64, 64, 1),
         "dec1": (24, 40, 640, 256, 1), "dec0": (12, 20, 768, 512, 1)}
G = torch.tensor([[1., 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])


def put(ue, addr, t):
    put_bytes(ue, addr, user_dma_core.tensor_le_bytes(t.to(torch.bfloat16).contiguous()))


def put_bytes(ue, addr, data):
    off = 0
    while off < len(data):
        n = min(1 << 20, len(data) - off)
        ue.dma_write(DMA_DEVICE_H2C, addr + off, data[off:off + n], n)
        off += n


def get(ue, addr, numel):
    buf = bytearray(numel * 2)
    off = 0
    while off < len(buf):
        n = min(1 << 20, len(buf) - off)
        part = bytearray(n)
        ue.dma_read(DMA_DEVICE_C2H, addr + off, part, n)
        buf[off:off + n] = part
        off += n
    return user_dma_core.tensor_from_le_bytes(buf, torch.bfloat16).float()


def scopy(e, src, dst, count, chunk, src_jump=0, dst_jump=0):
    """stereo_ess_test._scopy: strided read -> URAM A -> strided write, SRAM-sized pieces."""
    src_jump = 0 if src_jump == chunk else src_jump
    dst_jump = 0 if dst_jump == chunk else dst_jump
    per = max(1, STAGE // chunk)
    i0 = 0
    while i0 < count:
        n = min(per, count - i0)
        e.accelerator_memory_to_sram(src + i0 * (src_jump or chunk), 0, n * chunk // 2,
                                     stride_bytes_per_chunk=chunk if src_jump else 0, stride_jump_bytes=src_jump)
        e.sram_to_accelerator_memory(0, dst + i0 * (dst_jump or chunk), n * chunk // 2,
                                     stride_bytes_per_chunk=chunk if dst_jump else 0, stride_jump_bytes=dst_jump)
        i0 += n


class Map:
    def __init__(self, H, W, C, halo, addr):
        self.H, self.W, self.C, self.halo, self.addr = H, W, C, halo, addr
        self.pb = C * 2
        self.Wp, self.Hp = W + 2 * halo, H + 2 * halo
        self.pitch = self.Wp * self.pb
        self.npix = self.Hp * self.Wp + 2 * self.Wp + 320

    def pix(self, y, x):
        return self.addr + ((y + self.halo) * self.Wp + x + self.halo) * self.pb


def split(n, i, ne):
    base, rem = divmod(n, ne)
    a = i * base + min(i, rem)
    return a, a + base + (1 if i < rem else 0)


def quantize_if8_channels(w: torch.Tensor):
    """[OC, C, 3, 3] fp32 (C, OC multiples of 64) -> codes [OC, C, 3, 3] uint8, scales [OC, 9*ct] bf16, dequant."""
    OC, C, kh, kw = w.shape
    ct = C // V
    rows = w.float().view(OC, ct, V, kh, kw).permute(0, 3, 4, 1, 2).reshape(OC, kh * kw * ct * V)
    data, sc = quant_lib.quantize_if8(rows, block_size=V, int_variant=None)
    codes = (torch.frombuffer(bytearray(data), dtype=torch.uint8).view(OC, kh, kw, ct, V)
             .permute(0, 3, 4, 1, 2).reshape(OC, C, kh, kw).contiguous())
    scales = torch.frombuffer(bytearray(sc), dtype=torch.bfloat16).view(OC, kh * kw * ct).clone()
    deq = quant_lib.dequantize_if8(data, sc, OC, kh * kw * ct * V, block_size=V)
    deq = (torch.as_tensor(deq).float().view(OC, kh, kw, ct, V).permute(0, 3, 4, 1, 2)
           .reshape(OC, C, kh, kw).contiguous())
    return codes, scales, deq


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--cases", default="dec3,enc1,dec1,dec0")
    ap.add_argument("--paths", default="direct,winograd,native16,native8")
    ap.add_argument("--reps", type=int, default=3, help="convs per program (time = median run / reps)")
    args = ap.parse_args()
    set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    print(user_dma_core.hardware_info_summary())
    NE = args.engines
    assert NE <= user_dma_core.ANDROMEDA_CORE_COUNT
    paths = args.paths.split(",")
    windows, model_base = model_multicore_layout(NE, model_bytes=1 << 30)
    ue = UnifiedEngine(params_dram_base=model_base, tensor_dram_base=model_base + 0x1000_0000,
                       program_dram_base=model_base + 0x3000_0000, conv_geometry_mode=CONV_GEOMETRY_QUEUE_CONFIG)
    ue.software_reset()
    arena = PrivateArena(NE, windows=windows, tensor_bytes=0x1000_0000)
    sched = MultiEngineScheduler(ue, num_engines=NE, handshake="four_phase", region_rendezvous="master_worker",
                                 arena=arena)
    for w in sched.workers:                                 # CONV2D geometry as in-program CONFIG on every engine
        w.conv_geometry_mode = CONV_GEOMETRY_QUEUE_CONFIG
    sched.preclear_flags()
    for i in range(NE):
        arena.alloc_weights(i, 65536, "selftest_guard")
    al = lambda n: ue.allocate_tensor_dram((n + 127) // 128 * 128)
    I64 = al(64 * 64 * 2)
    put(ue, I64, torch.eye(64))
    ZEROS = al(STAGE)
    put(ue, ZEROS, torch.zeros(STAGE // 2))
    peak = NE * V * 2 * 1e3 / user_dma_core.CLOCK_CYCLE_TIME_NS / 1e9      # GFLOPS, 64 MAC/cycle/engine

    def pal(i, n, what):                                    # private, zero-filled
        n = (n + 127) // 128 * 128
        a = arena.alloc_tensor(i, n, what)
        put(ue, a, torch.zeros(n // 2))
        return a

    def pwt(i, t, what):                                    # private weight replica (bf16 tensor)
        a = arena.alloc_weights(i, (t.numel() * 2 + 127) // 128 * 128, what)
        put(ue, a, t)
        return a

    def pwb(i, data, what):                                 # private weight replica (raw bytes)
        a = arena.alloc_weights(i, (len(data) + 127) // 128 * 128, what)
        put_bytes(ue, a, data)
        return a

    lalu_relu = user_dma_core._conv_fused_lalu(True, False, False)
    torch.manual_seed(0)
    ok = True
    summary = []
    for name in args.cases.split(","):
        H, W, Cin, Cout, halo = CASES[name]
        TW = W // 2
        K = 9 * Cin
        nominal = 2.0 * H * W * K * Cout                                # FLOPs of the conv as written
        pick = select_conv_path(kernel_h=3, kernel_w=3, stride_s=1, dilation=1, c_in=Cin, oc_count=Cout,
                                data_type=TYPE.FLOAT16)
        print(f"== {name} {H}x{W} {Cin}->{Cout}: select_conv_path -> {pick!r} "
              f"(CONV3X3_BF16_FALLBACK={user_dma_core.CONV3X3_BF16_FALLBACK!r})", flush=True)
        x = torch.relu(torch.randn(Cin, H, W))
        g = torch.randn(Cout, Cin, 3, 3) * (1.0 / K ** 0.5)
        b = torch.randn(Cout) * 0.1
        xq, gq, bq = (t.to(torch.bfloat16).float() for t in (x, g, b))
        ref = torch.relu(F.conv2d(xq[None], gq, bq, padding=1))[0]
        Wd = gq.permute(0, 2, 3, 1).reshape(Cout, K)                     # direct: [Cout, (ky, kx, cin)]
        U = torch.einsum("ik,ockl,jl->ocij", G, gq, G)                   # Winograd: U_xi = (G g G^T)[:, :, r, c]
        sgn = lambda r, c: (-1.0 if r == 2 else 1.0) * (-1.0 if c == 2 else 1.0)
        codes8, scales8, deq8 = quantize_if8_channels(gq)
        ref8 = torch.relu(F.conv2d(xq[None], deq8, bq, padding=1))[0]
        # shared maps
        X = Map(H, W, Cin, halo, 0)
        X.addr = al(X.npix * X.pb)
        grid = torch.zeros(X.npix, Cin)
        grid[:X.Hp * X.Wp].view(X.Hp, X.Wp, Cin)[halo:halo + H, halo:halo + W] = xq.permute(1, 2, 0)
        put(ue, X.addr, grid)
        Ys = {}
        for p in paths:
            m = Map(H, W, Cout, halo, 0)
            m.addr = al(m.npix * m.pb)
            put(ue, m.addr, torch.zeros(m.npix * Cout))
            Ys[p] = m
        # native plans: one per (format, band rows); tiles are in the band's own padded coordinates
        plans = {}
        for fmt, limit, dtype in (("native16", 4096, TYPE.FLOAT16), ("native8", 8192, TYPE.IF8)):
            if fmt not in paths:
                continue
            bpb = 128 if dtype == TYPE.FLOAT16 else 64
            per_rows = {}
            for i in range(NE):
                ya, yb = split(H, i, NE)
                if yb <= ya or (yb - ya) in per_rows:
                    continue
                out_h, out_w, occ, tiles = user_dma_core.plan_conv2d_layer_tiles(
                    c_in=Cin, oc_count=Cout, in_h=yb - ya, in_w=W, kernel_h=3, kernel_w=3, stride_s=1, pad=1,
                    bias_enabled=True, weight_block_limit=limit, act_uram_addr=0, wb_uram_addr=ACT_URAM_LINES)
                assert (out_h, out_w) == (yb - ya, W)
                assert occ * 9 * (Cin // V) * bpb <= URAM_B_BYTES, (name, fmt, occ)
                per_rows[yb - ya] = (occ, tiles, user_dma_core.conv2d_tile_geometry_groups(tiles))
            occ = next(iter(per_rows.values()))[0]
            assert all(v[0] == occ for v in per_rows.values()), (name, fmt, per_rows)
            chunks = []
            for oc0 in range(0, Cout, occ):
                sl = slice(oc0, oc0 + occ)
                if dtype == TYPE.FLOAT16:
                    wblk = user_dma_core.conv2d_pack_weight_stream(gq[sl], 1, 1, TYPE.FLOAT16)
                    sblk = None
                else:
                    wblk = user_dma_core.conv2d_pack_weight_stream(codes8[sl], 1, 1, TYPE.IF8)
                    sblk = user_dma_core.conv2d_pack_scale_stream(scales8[sl], occ, 9 * (Cin // V), 1, 1)
                bblk = user_dma_core.conv2d_pack_bias_stream(bq[sl].to(torch.bfloat16), 1, 1)
                chunks.append((oc0, wblk.numpy().tobytes(),
                               None if sblk is None else user_dma_core.tensor_le_bytes(sblk.contiguous()),
                               user_dma_core.tensor_le_bytes(bblk)))
            plans[fmt] = (dtype, occ, per_rows, chunks)
            print(f"   {fmt}: oc chunk {occ} ({Cout // occ} chunks), pattern {occ * 9 * (Cin // V) * bpb // 1024} KB, "
                  f"tiles per band {[len(v[1]) for v in per_rows.values()]}", flush=True)
        # per-engine private state
        P = []
        for i in range(NE):
            ya, yb = split(H, i, NE)
            ta, tb = split(H // 2, i, NE)
            nt = (tb - ta) * TW
            st = {"rows": (ya, yb), "tiles": (ta, tb),
                  "bd": pwt(i, bq, "bias"), "wd": pwt(i, Wd, "w_direct"),
                  "us": [pwt(i, (sgn(r, c) * U[:, :, r, c]).contiguous(), f"us{r}{c}") for r in range(4) for c in range(4)],
                  "im": pal(i, max(1, (yb - ya)) * X.Wp * K * 2, "im2col")}
            st["D"] = [pal(i, max(nt, 1) * Cin * 2, f"D{p}") for p in range(16)]
            st["M"] = [pal(i, max(nt, 1) * Cout * 2, f"M{p}") for p in range(16)]
            for fmt, (dtype, occ, per_rows, chunks) in plans.items():
                st[fmt] = [(oc0, pwb(i, wb, f"{fmt}_w{oc0}"), None if sb is None else pwb(i, sb, f"{fmt}_s{oc0}"),
                            pwb(i, bb, f"{fmt}_b{oc0}")) for oc0, wb, sb, bb in chunks]
            P.append(st)

        def direct(e, i):
            st = P[i]
            ya, yb = st["rows"]
            if yb <= ya:
                return 0
            Yd = Ys["direct"]
            gM = e.alloc_isa_reg()
            Mr = (yb - ya) * X.Wp
            p0 = ya * X.Wp
            for ky in range(3):                                  # row3 im2col, as conv2d
                scopy(e, X.addr + (p0 + (halo + ky - 1) * X.Wp + halo - 1) * X.pb, st["im"] + ky * 3 * Cin * 2,
                      Mr, 3 * X.pb, src_jump=X.pb, dst_jump=K * 2)
            e.generate_instruction_add_set(gM, Mr)
            e.matmat_mul_core(M=Mr, K=K, N=Cout, A_DRAM_ADDR=st["im"], B_DRAM_ADDR=st["wd"],
                              OUTPUT_DRAM_ADDR=Yd.pix(ya, 0), C_DRAM_ADDR=st["bd"], bias_mode="broadcast_N",
                              gpr_M_reg=gM, clamp_enable=True, clamp_min=0.0, clamp_max=float("inf"))
            chunk, jump = 2 * halo * Yd.pb, Yd.Wp * Yd.pb             # halo re-zero (own rows)
            e.accelerator_memory_to_sram(ZEROS, 0, (yb - ya) * chunk // 2)
            e.sram_to_accelerator_memory(0, Yd.pix(ya, W), (yb - ya) * chunk // 2, stride_bytes_per_chunk=chunk,
                                         stride_jump_bytes=jump)
            return 1

        def winograd(e, i):
            """winograd_probe 'weo': even/odd input stage, transforms in SRAM, 16 matmuls, output transform in SRAM."""
            st = P[i]
            ta, tb = st["tiles"]
            if tb <= ta:
                return 0
            Yw = Ys["winograd"]
            nt = (tb - ta) * TW
            gM = e.alloc_isa_reg()
            D, Mx = st["D"], st["M"]
            Pout = TW * Cout * 2
            assert 18 * Pout <= 0x80000 and Pout % 128 == 0
            A0, B0 = 0x00000, 0x80000
            add = lambda a, b_, o, n: e.eltwise_add_core(a, b_, o, n)
            sub = lambda a, b_, o, n: e.eltwise_sub_core(a, b_, o, n)
            nel = TW * Cin
            up = lambda n: (n + 127) // 128 * 128
            sz = up((TW + 1) * X.pb)
            assert 12 * sz <= 0x80000, (name, sz)
            DS = lambda r, par: (A0 + (r * 2 + par) * sz) if r < 2 else (B0 + ((r - 2) * 2 + par) * sz)
            TA = lambda k, par: A0 + (4 + k * 2 + par) * sz
            TB = lambda k, par: B0 + (4 + k * 2 + par) * sz
            ne = (TW + 1) * Cin
            for ty in range(ta, tb):
                off = (ty - ta) * TW
                for r in range(4):                               # each source row once: even / odd pixels
                    for par in (0, 1):
                        e.accelerator_memory_to_sram(X.pix(2 * ty - 1 + r, -1 + par), DS(r, par), ne,
                                                     stride_bytes_per_chunk=X.pb, stride_jump_bytes=2 * X.pb)
                for T in (TA, TB):                               # rows: T0=D0-D2, T1=D1+D2, -T2=D1-D2, T3=D1-D3
                    for par in (0, 1):
                        sub(DS(0, par), DS(2, par), T(0, par), ne)
                        add(DS(1, par), DS(2, par), T(1, par), ne)
                        sub(DS(1, par), DS(2, par), T(2, par), ne)
                        sub(DS(1, par), DS(3, par), T(3, par), ne)
                for k in range(4):                               # cols on tile windows (contiguous slices)
                    Vs = [A0 + q * sz for q in range(4)]
                    sub(TA(k, 0), TB(k, 0) + X.pb, Vs[0], nel)
                    add(TA(k, 1), TB(k, 0) + X.pb, Vs[1], nel)
                    sub(TA(k, 1), TB(k, 0) + X.pb, Vs[2], nel)
                    sub(TA(k, 1), TB(k, 1) + X.pb, Vs[3], nel)
                    for q in range(4):
                        e.sram_to_accelerator_memory(Vs[q], D[k * 4 + q] + off * Cin * 2, nel)
            for xi in range(16):                                 # 16 matmuls on the (signed) V planes
                e.generate_instruction_add_set(gM, nt)
                kw = dict(C_DRAM_ADDR=st["bd"], bias_mode="broadcast_N") if xi == 5 else {}
                e.matmat_mul_core(M=nt, K=Cin, N=Cout, A_DRAM_ADDR=D[xi], B_DRAM_ADDR=st["us"][xi],
                                  OUTPUT_DRAM_ADDR=Mx[xi], gpr_M_reg=gM, **kw)
            nel = TW * Cout
            Msr = lambda r, c: A0 + c * Pout if r == 1 else B0 + ({0: 0, 2: 1, 3: 2}[r] * 4 + c) * Pout
            Ssr = lambda i_, c: A0 + (4 + i_) * Pout if c == 1 else B0 + (12 + i_ * 3 + {0: 0, 2: 1, 3: 2}[c]) * Pout
            T0, T1 = A0 + 6 * Pout, A0 + 7 * Pout
            Ysr = lambda k_: A0 + (8 + k_) * Pout
            Y = Mx[:4]
            for ty in range(ta, tb):
                off = (ty - ta) * TW
                for r in range(4):
                    for c in range(4):
                        e.accelerator_memory_to_sram(Mx[r * 4 + c] + off * Cout * 2, Msr(r, c), nel)
                for c in range(4):                               # S0 = (M1 + M2) + M0, S1 = (M1 - M2) - M3
                    add(Msr(1, c), Msr(2, c), T0, nel)
                    add(T0, Msr(0, c), Ssr(0, c), nel)
                    sub(Msr(1, c), Msr(2, c), T1, nel)
                    sub(T1, Msr(3, c), Ssr(1, c), nel)
                for i_ in range(2):                              # y0 = (S1 + S2) + S0, y1 = (S1 - S2) - S3
                    add(Ssr(i_, 1), Ssr(i_, 2), T0, nel)
                    add(T0, Ssr(i_, 0), Ysr(2 * i_), nel)
                    sub(Ssr(i_, 1), Ssr(i_, 2), T1, nel)
                    sub(T1, Ssr(i_, 3), Ysr(2 * i_ + 1), nel)
                for k_ in range(4):
                    e.sram_to_accelerator_memory(Ysr(k_), Y[k_] + off * Cout * 2, nel)
            ro = nt * Cout // 64
            for yp in Y:                                         # ReLU (identity-matmul clamp)
                e.generate_instruction_add_set(gM, ro)
                e.activation_core(M=ro, N=64, A_DRAM_ADDR=yp, OUTPUT_DRAM_ADDR=yp, IDENTITY_DRAM_ADDR=I64,
                                  activation="clamp", gpr_M_reg=gM)
            for i_ in range(2):
                for j in range(2):
                    for ty in range(ta, tb):
                        scopy(e, Y[2 * i_ + j] + (ty - ta) * TW * Cout * 2, Yw.pix(2 * ty + i_, j), TW, Cout * 2,
                              dst_jump=2 * Yw.pb)
            return 16

        def native(e, i, fmt):
            """CONV2D with weight_reuse over this engine's row band: window from the halo map -> URAM A,
            one launch per (oc chunk, tile), writeback scattered into the destination map."""
            st = P[i]
            ya, yb = st["rows"]
            if yb <= ya:
                return 0
            Yn = Ys[fmt]
            dtype, occ, per_rows, _ = plans[fmt]
            _, tiles, groups = per_rows[yb - ya]
            ct = Cin // V
            taps = 9 * ct
            mode, la, lb = lalu_relu
            launches = 0
            for oc0, w_addr, s_addr, b_addr in st[fmt]:
                for start, stop, th, tw, win_h, win_w in groups:
                    if s_addr is not None:
                        e.accelerator_memory_to_scale_sram(s_addr, occ * taps)
                    e.accelerator_memory_to_bias_sram(b_addr, occ)
                    for tile in tiles[start:stop]:
                        oy, ox, th_, tw_, y0, x0, wh, ww = (int(v) for v in tile)
                        a = X.pix(ya + y0 - 1, x0 - 1)                    # padded-window origin in the halo map
                        row = ww * X.pb
                        if row == X.pitch:
                            e.accelerator_memory_to_sram(a, 0, wh * row // 2)
                        else:
                            e.accelerator_memory_to_sram(a, 0, wh * row // 2, stride_bytes_per_chunk=row,
                                                         stride_jump_bytes=X.pitch)
                        e.start_queue_for_conv2d_operation(
                            act_sram_start_addr=0, output_sram_wb_addr=WB_SRAM, weights_dram_addr=w_addr,
                            kernel_w=3, kernel_h=3, ct=ct, oc_count=occ, out_w=tw_, out_h=th_, w_pad=ww,
                            stride_s=1, data_type=dtype, bias_enable=True, lalu_mode=mode, lalu_a=la, lalu_b=lb,
                            dilation=1, gather=False, c_in=Cin, weight_reuse=True)
                        launches += 1
                        oc_b = occ * 2
                        if oc_b == Yn.pb:                                 # whole pixels per row
                            row = tw_ * Yn.pb
                            if th_ == 1 or row == Yn.pitch:
                                e.sram_to_accelerator_memory(WB_SRAM, Yn.pix(ya + oy, ox), th_ * row // 2)
                            else:
                                e.sram_to_accelerator_memory(WB_SRAM, Yn.pix(ya + oy, ox), th_ * row // 2,
                                                             stride_bytes_per_chunk=row, stride_jump_bytes=Yn.pitch)
                        else:                                             # oc chunk at its lane offset, per row
                            for r in range(th_):
                                e.sram_to_accelerator_memory(WB_SRAM + r * tw_ * oc_b, Yn.pix(ya + oy + r, ox) + oc0 * 2,
                                                             tw_ * oc_b // 2, stride_bytes_per_chunk=oc_b,
                                                             stride_jump_bytes=Yn.pb)
            return launches

        def run(body):
            ue.reset_program_dram_addr()
            for w in sched.workers:
                w.reset_program_dram_addr()
            ue.start_capture()
            ue.reset_isa_reg_counter()
            ue.reset_inst_ptr_counter()
            sched.begin_program()
            launches = [0] * NE
            for _ in range(args.reps):
                def _b(ctx):
                    launches[ctx.engine_idx] = body(ctx.unsafe_ue, ctx.engine_idx)
                sched.sharded_region(64 * NE, _b)
            progs = sched.finalize()
            ue.stop_capture()
            ue.generate_instruction_halt()
            addr = ue.get_program_dram_addr()
            ue.write_captured_instructions_to_dram(addr)
            ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
            ninst = ue.capture_count
            ue.clear_capture_buffer()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                sched.start_workers(progs)
                ue.start_execute_from_dram(addr)
                ue.wait_queue(120.0)
                ts.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.5)
            return sorted(ts)[1] / args.reps, ninst, max(launches)

        bodies = {"direct": direct, "winograd": winograd,
                  "native16": lambda e, i: native(e, i, "native16"), "native8": lambda e, i: native(e, i, "native8")}
        res = {}
        for meth in paths:
            print(f"  {name:5s} {meth:9s} compiling + running ...", flush=True)
            ms, ninst, nl = run(bodies[meth])
            ym = Ys[meth]
            hw = get(ue, ym.addr, ym.Hp * ym.Wp * Cout).view(ym.Hp, ym.Wp, Cout)[halo:halo + H, halo:halo + W]
            hw = hw.permute(2, 0, 1)
            snr = float(calculate_snr(ref.flatten(), hw.flatten()))
            extra = ""
            if meth == "native8":
                snr8 = float(calculate_snr(ref8.flatten(), hw.flatten()))
                extra = f"  (vs dequant IF8 ref {snr8:6.2f} dB)"
                good_snr = snr8 >= 30
            else:
                good_snr = snr >= 30
            bad = int((~torch.isfinite(hw)).sum())
            good = good_snr and bad == 0
            ok &= good
            gf = nominal / (ms * 1e-3) / 1e9
            res[meth] = ms
            summary.append((name, meth, ms, gf, snr))
            print(f"  {name:5s} {H}x{W} {Cin}->{Cout}  {meth:9s} {ms:8.3f} ms/conv  {gf:7.1f} GFLOPS nominal "
                  f"({100 * gf / peak:5.1f}% of {peak:.0f} peak)  {nl} launches/eng  {ninst} primary instr  "
                  f"SNR {snr:6.2f} dB{extra}  non-finite {bad}  {'PASS' if good else 'FAIL'}", flush=True)
        if "direct" in res:
            order = sorted(res, key=res.get)
            print(f"  {name:5s} order: " + " < ".join(f"{m} {res[m]:.2f}" for m in order) +
                  "   |  " + "  ".join(f"{m} {res['direct'] / res[m]:.2f}x vs direct" for m in order if m != "direct"),
                  flush=True)
        ue.reset_tensor_dram_addr()
        al(64 * 64 * 2 + STAGE)                                  # keep I64 / ZEROS at the same addresses
    print()
    print(f"{'case':6s} {'path':9s} {'ms/conv':>9s} {'GFLOPS':>8s} {'%peak':>6s} {'SNR':>7s}")
    for name, meth, ms, gf, snr in summary:
        print(f"{name:6s} {meth:9s} {ms:9.3f} {gf:8.1f} {100 * gf / peak:6.1f} {snr:7.2f}")
    if {"direct", "native16"} <= set(paths):
        d = [m for n, m, *_ in summary if m == "direct"]
        print("\nbf16 fallback verdict per case (native16 faster than direct?):")
        for n in args.cases.split(","):
            t = {m: ms for nn, m, ms, *_ in summary if nn == n}
            if "direct" in t and "native16" in t:
                print(f"  {n:6s} native16 {t['native16']:.3f} ms vs direct {t['direct']:.3f} ms -> "
                      f"{'walker' if t['native16'] < t['direct'] else 'im2col'}")
        print("set user_dma_core.CONV3X3_BF16_FALLBACK to the winner.")
    print("CONV3X3_PATHS_PROBE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
