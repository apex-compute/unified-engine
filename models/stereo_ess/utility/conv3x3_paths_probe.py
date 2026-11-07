#!/usr/bin/env python3
"""Board probe: Winograd vs the native CONV2D walker vs im2col, same shapes, same engines, same input.

    python models/stereo_ess/utility/conv3x3_paths_probe.py                       # light_ess cases, 8 engines
    python models/stereo_ess/utility/conv3x3_paths_probe.py --cases dec3 --engines 1
    python models/stereo_ess/utility/conv3x3_paths_probe.py --paths direct,native16
    python models/stereo_ess/utility/conv3x3_paths_probe.py --sweep               # conv types x shapes, writes graphs
    python models/stereo_ess/utility/conv3x3_paths_probe.py --sweep --spatial 24x40,96x160 --channels 128,512
    python models/stereo_ess/utility/conv3x3_paths_probe.py --plot-only scratchpad/conv_paths_sweep_e8.json

Question: for which convs does Winograd F(2x2,3x3) actually win, and by how much, now that the CONV2D walker
no longer re-streams its weights per output pixel (927fd8d RTL, UE_CONV_CTRL[30] weight_reuse)? The named cases
set user_dma_core.CONV3X3_BF16_FALLBACK; --sweep maps where Winograd pays off and where it does not apply.

Paths (each engine works on its own band of output rows, like the stereo --engines 8 program):
  direct     im2col in DRAM (strided row copies) + one runtime-M bf16 matmul, K = kh*kw*Cin, bias + CLAMP ReLU.
  winograd   F(2x2,3x3) with both transforms in SRAM (the winograd_probe 'weo' flow): each source row loaded once
             as even / odd arrays, B^T d B as SRAM add/subs, 16 bf16 matmuls K = Cin, A^T M A in SRAM, ReLU, scatter.
             Tile rows wider than the SRAM staging area are processed in column chunks. Only 3x3 / stride 1 /
             dilation 1; every other conv is "n/a". The model's fully fused flow (U resident in URAM B, matvec per
             tile) is faster still; this is the standalone upper bound on its cost.
  native16   hardware CONV2D opcode, FLOAT16 (bf16 storage) weights with weight_reuse: the conv's [oc][tap]
             pattern (128 B blocks) is loaded into URAM B once per launch and replayed for every tile pixel.
             Window staged from the halo map into URAM A, bias + CLAMP ReLU fused, writeback scattered to the map.
  native8    same launch with IF8 weights (64 B blocks, MixMSE INT8/FP8 per block + scale BRAM) -- the format
             flux2's VAE runs. Its SNR is reported against the bf16 reference AND against the dequantised weights.

Every path computes relu(conv(x) + b) with "same" padding on the same bf16 input and is checked against torch fp32
on the bf16 data. Reported per path: ms per conv (median of 3 runs / reps), effective GFLOPS on the nominal
2*Hout*Wout*kh*kw*Cin*Cout (Winograd executes 2.25x fewer MACs, so it can read above the matmul peak), SNR,
launches per engine. A path that cannot be built for a shape (SRAM / URAM / arena limits) is reported as "skip".

Named cases (the real light_ess shapes, as winograd_probe): dec3 96x160 192->64 | enc1 48x80 64->64 |
dec1 24x40 640->256 | dec0 12x20 768->512; flux2_klein_4b VAE: vae512 128x128 512->512 | vae256 256x256
256->256 | vae128 512x512 128->128 (Winograd works through these in tile-row groups, WINO_BUDGET per engine).
--auto: UnifiedEngine.conv2d_auto check -- ROUTE_TABLE on the host, then AUTO_CASES on the board (1 engine).
--sweep: every conv type (1x1, 3x3, 3x3 s2, 3x3 d2, 5x5, 7x7 s2) at every input map size (--spatial, H x W)
and channel count (--channels; Cin = Cout, so "128->128" is 128 input and 128 output channels). Results go to
<--out>_e<N>.json, <--out>_e<N>.png (speedup vs native16, one panel per conv type, one bar group per shape) and
<--out>_e<N>_snr.png (speed vs SNR for every measured conv).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

STAGE = 4000 * 128
V = 64
ACT_URAM_LINES = 0x300                 # padded window in URAM A lines [0, 0x300), writeback from 0x300
WB_SRAM = ACT_URAM_LINES << 7
URAM_B_BYTES = 512 * 1024
SRAM_BANK = 0x80000
# name: (H, W, Cin, Cout, halo)
CASES = {"dec3": (96, 160, 192, 64, 2), "enc1": (48, 80, 64, 64, 1),
         "dec1": (24, 40, 640, 256, 1), "dec0": (12, 20, 768, 512, 1),
         # flux2_klein_4b VAE decoder at 512x512 (resnet conv1/conv2 per up-block)
         "vae512": (128, 128, 512, 512, 1), "vae256": (256, 256, 256, 256, 1), "vae128": (512, 512, 128, 128, 1)}
SWEEP_SPATIAL = "12x20,24x40,48x80,96x160"
SWEEP_CHANNELS = "64,128,256,512"
IM2COL_BUDGET = 64 << 20               # bytes of im2col per matmul; bigger bands are done in row chunks
WINO_BUDGET = 128 << 20                # bytes of Winograd D + M planes per engine; bigger bands go in row groups
# label: (kernel, stride, dilation)
CONV_TYPES = (("1x1", 1, 1, 1), ("3x3", 3, 1, 1), ("3x3 s2", 3, 2, 1), ("3x3 d2", 3, 1, 2),
              ("5x5", 5, 1, 1), ("7x7", 7, 1, 1), ("7x7 s2", 7, 2, 1))
WINO_NOTE = {"plain": "Winograd F(2x2,3x3) directly",
             "subgrid": "Winograd via 4 parity sub-grids (each a 3x3 conv on a quarter map)",
             "split": "Winograd via {n} 3x3 kernel pieces summed in the M planes"}
PATH_ORDER = ("winograd", "native16", "direct", "native8")
# --auto: conv2d_auto on the board, 1 engine, 12x20 maps (small enough for its single-launch walker path).
# (name, k, stride, dilation, Cin, Cout, dtype, allow_winograd, expected route) -- expectations from the
# --sweep result on 927fd8d: Winograd wins 3x3/s1/d1 from 128 channels, native CONV2D everywhere else.
AUTO_CASES = (
    ("3x3 128->128", 3, 1, 1, 128, 128, "bf16", True, "winograd"),
    ("3x3 128->128 vetoed", 3, 1, 1, 128, 128, "bf16", False, "walker"),
    ("3x3 64->64", 3, 1, 1, 64, 64, "bf16", True, "walker"),
    ("1x1 128->128", 1, 1, 1, 128, 128, "bf16", True, "matmul"),
    ("3x3 s2 128->128", 3, 2, 1, 128, 128, "bf16", True, "walker"),
    ("3x3 d2 128->128", 3, 1, 2, 128, 128, "bf16", True, "walker"),
    ("3x3 128->128 IF8", 3, 1, 1, 128, 128, "if8", True, "walker"),
)
# Routing rule only (host): real model layers from the code sweep, incl. shapes with no board lowering here.
# (layer, k, stride, dilation, Cin, Cout, dtype, groups, allow_winograd, expected)
ROUTE_TABLE = (
    ("flux2 VAE resnet 3x3 512->512", 3, 1, 1, 512, 512, "bf16", 1, True, "winograd"),
    ("flux2 VAE resnet 3x3 256->256", 3, 1, 1, 256, 256, "bf16", 1, True, "winograd"),
    ("flux2 VAE resnet 3x3 128->128", 3, 1, 1, 128, 128, "bf16", 1, True, "winograd"),
    ("flux2 VAE conv_in 3x3 32->512", 3, 1, 1, 32, 512, "bf16", 1, True, "walker"),
    ("flux2 VAE attention 1x1 512->512", 1, 1, 1, 512, 512, "bf16", 1, True, "matmul"),
    ("ACT layer1 3x3 64->64", 3, 1, 1, 64, 64, "bf16", 1, True, "walker"),
    ("ACT layer2 3x3 128->128", 3, 1, 1, 128, 128, "bf16", 1, True, "winograd"),
    ("ACT layer4 3x3 512->512", 3, 1, 1, 512, 512, "bf16", 1, True, "winograd"),
    ("ACT layer2 3x3 s2 64->128", 3, 2, 1, 64, 128, "bf16", 1, True, "walker"),
    ("mobilesam neck 3x3 256->256", 3, 1, 1, 256, 256, "bf16", 1, True, "winograd"),
    ("light_ess dec1 3x3 640->256", 3, 1, 1, 640, 256, "bf16", 1, True, "winograd"),
    ("threshold 3x3 96->96", 3, 1, 1, 96, 96, "bf16", 1, True, "winograd"),
    ("3x3 d2 256->256", 3, 1, 2, 256, 256, "bf16", 1, True, "walker"),
    ("5x5 256->256", 5, 1, 1, 256, 256, "bf16", 1, True, "walker"),
    ("veto 3x3 512->512 allow_winograd=False", 3, 1, 1, 512, 512, "bf16", 1, False, "walker"),
    ("yolov5s 3x3 128->128 IF4", 3, 1, 1, 128, 128, "if4", 1, True, "walker"),
    ("unet 3x3 256->256 IF8", 3, 1, 1, 256, 256, "if8", 1, True, "walker"),
    ("mobilenetv2 depthwise 3x3 C=144", 3, 1, 1, 144, 144, "bf16", 144, True, "grouped"),
    ("mobilesam depthwise 3x3 C=256", 3, 1, 1, 256, 256, "bf16", 256, True, "grouped"),
    ("SigLIP patch embed 14x14 s14 3->1152", 14, 14, 1, 3, 1152, "bf16", 1, True, "matmul"),
    ("swin patch embed 4x4 s4 3->192", 4, 4, 1, 3, 192, "bf16", 1, True, "matmul"),
)


def check_routes():
    """select_conv_path on ROUTE_TABLE (host only, no board). Returns True when every route matches."""
    import user_dma_core
    from user_dma_core import TYPE, select_conv_path
    types = {"bf16": TYPE.FLOAT16, "if8": TYPE.IF8, "if4": TYPE.IF4}
    ok = True
    print("routing rule (select_conv_path, host):")
    for name, k, s, d, cin, cout, dt, groups, allow, expect in ROUTE_TABLE:
        got = select_conv_path(kernel_h=k, kernel_w=k, stride_s=s, dilation=d, c_in=cin, oc_count=cout,
                               data_type=types[dt], allow_winograd=allow, groups=groups)
        ok &= got == expect
        print(f"  {'PASS' if got == expect else 'FAIL'}  {name:40s} expected {expect:8s} got {got}")
    # 7x7 s2 512->512 bf16: the sweep's native launch has 8 oc = 16 B per pixel, not scatterable
    fits = user_dma_core.conv_walker_pattern_fits(c_in=512, oc_count=512, kernel_h=7, kernel_w=7,
                                                  data_type=TYPE.FLOAT16)
    ok &= not fits
    print(f"  {'PASS' if not fits else 'FAIL'}  {'7x7 512->512 bf16 walker does not fit':40s} expected False    "
          f"got {fits}")
    return ok


class Spec:
    def __init__(self, name, group, H, W, cin, cout, k=3, s=1, d=1, halo=1):
        self.name, self.group = name, group
        self.H, self.W, self.Cin, self.Cout, self.k, self.s, self.d = H, W, cin, cout, k, s, d
        self.pad = d * (k - 1) // 2
        self.halo = max(halo, self.pad, 1)
        self.Hout, self.Wout = (H - 1) // s + 1, (W - 1) // s + 1
        if self.winograd_mode == "split":               # the padded 3x3 pieces read past the conv's own padding
            n3 = -(-k // 3)
            self.halo = max(self.halo, 3 * (n3 - 1) - self.pad + 2)

    @property
    def kind(self):
        return f"{self.k}x{self.k}" + (f" s{self.s}" if self.s > 1 else "") + (f" d{self.d}" if self.d > 1 else "")

    @property
    def winograd_mode(self):
        """How F(2x2,3x3) reaches this conv: "plain" (3x3), "subgrid" (3x3 dilation 2 = four 3x3 convs on the
        parity sub-grids), "split" (5x5 / 7x7 = n*n 3x3 kernel pieces accumulated in the M planes), or None
        (stride 2: its polyphase pieces are 2x2 / 2x1 / 1x1 kernels, no F(2x2,3x3) in them)."""
        if self.s != 1:
            return None
        if self.k == 3 and self.d == 1 and self.H % 2 == 0 and self.W % 2 == 0:
            return "plain"
        if self.k == 3 and self.d == 2 and self.H % 4 == 0 and self.W % 4 == 0:
            return "subgrid"
        if self.k in (5, 7) and self.d == 1 and self.H % 2 == 0 and self.W % 2 == 0:
            return "split"
        return None

    @property
    def winograd_applies(self):
        return self.winograd_mode is not None

    def describe(self):
        return f"{self.name} {self.kind} {self.H}x{self.W} {self.Cin}->{self.Cout}"


def up128(n):
    return (n + 127) // 128 * 128


def wino_chunk(W, Cin, Cout):
    """Widest tile-row chunk whose staging fits one SRAM bank: 12 input-side and 18 output-side slots."""
    TW = W // 2
    for twc in range(TW, 0, -1):
        if 18 * up128(twc * Cout * 2) <= SRAM_BANK and 12 * up128((twc + 1) * Cin * 2) <= SRAM_BANK:
            return twc
    return None


# ----------------------------------------------------------------------------------------------- board helpers
def put(ue, addr, t):
    import torch
    import user_dma_core
    put_bytes(ue, addr, user_dma_core.tensor_le_bytes(t.to(torch.bfloat16).contiguous()))


def put_bytes(ue, addr, data):
    from user_dma_core import DMA_DEVICE_H2C
    off = 0
    while off < len(data):
        n = min(1 << 20, len(data) - off)
        ue.dma_write(DMA_DEVICE_H2C, addr + off, data[off:off + n], n)
        off += n


def get(ue, addr, numel):
    import torch
    import user_dma_core
    from user_dma_core import DMA_DEVICE_C2H
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


def quantize_if8_channels(w):
    """[OC, C, kh, kw] fp32 (C multiple of 64) -> codes [OC, C, kh, kw] uint8, scales [OC, kh*kw*ct] bf16, dequant."""
    import torch
    import quant_lib
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


# ----------------------------------------------------------------------------------------------- probe
class Probe:
    def __init__(self, args):
        import user_dma_core
        from user_dma_core import CONV_GEOMETRY_QUEUE_CONFIG, UnifiedEngine, set_dma_device
        from multi_engine_shard import MultiEngineScheduler, PrivateArena, model_multicore_layout
        import torch
        self.args = args
        set_dma_device(args.dev)
        user_dma_core.configure_clock_from_hardware()
        print(user_dma_core.hardware_info_summary())
        self.NE = NE = args.engines
        assert NE <= user_dma_core.ANDROMEDA_CORE_COUNT
        windows, model_base = model_multicore_layout(NE, model_bytes=1 << 30)
        self.ue = ue = UnifiedEngine(params_dram_base=model_base, tensor_dram_base=model_base + 0x1000_0000,
                                     program_dram_base=model_base + 0x3000_0000,
                                     conv_geometry_mode=CONV_GEOMETRY_QUEUE_CONFIG)
        ue.software_reset()
        self.arena = arena = PrivateArena(NE, windows=windows, tensor_bytes=0x1000_0000)
        self.sched = MultiEngineScheduler(ue, num_engines=NE, handshake="four_phase",
                                          region_rendezvous="master_worker", arena=arena)
        for w in self.sched.workers:                       # CONV2D geometry as in-program CONFIG on every engine
            w.conv_geometry_mode = CONV_GEOMETRY_QUEUE_CONFIG
        self.sched.preclear_flags()
        for i in range(NE):
            arena.alloc_weights(i, 65536, "selftest_guard")
        self.I64 = self.al(64 * 64 * 2)
        put(ue, self.I64, torch.eye(64))
        self.ZEROS = self.al(STAGE)
        put(ue, self.ZEROS, torch.zeros(STAGE // 2))
        self.peak = NE * V * 2 / user_dma_core.CLOCK_CYCLE_TIME_NS   # GFLOPS, 64 MAC/cycle/engine
        self.lalu_relu = user_dma_core._conv_fused_lalu(True, False, False)
        self._snap = (list(arena._weight_cursor), list(arena._tensor_cursor), len(arena._weight_allocs))

    def al(self, n):
        return self.ue.allocate_tensor_dram(up128(n))

    def restore_arena(self):
        w, t, n = self._snap
        self.arena._weight_cursor[:] = w
        self.arena._tensor_cursor[:] = t
        del self.arena._weight_allocs[n:]

    def pal(self, i, n, what, zero=False):                 # private tensor
        import torch
        n = up128(max(n, 128))
        a = self.arena.alloc_tensor(i, n, what)
        if zero:
            put(self.ue, a, torch.zeros(n // 2))
        return a

    def pwt(self, i, t, what):                             # private weight replica (bf16 tensor)
        a = self.arena.alloc_weights(i, up128(t.numel() * 2), what)
        put(self.ue, a, t)
        return a

    def pwb(self, i, data, what):                          # private weight replica (raw bytes)
        a = self.arena.alloc_weights(i, up128(len(data)), what)
        put_bytes(self.ue, a, data)
        return a

    # ------------------------------------------------------------------------------------------- one case
    def run_case(self, sp, paths, auto=None):
        import torch
        import torch.nn.functional as F
        import user_dma_core
        from user_dma_core import TYPE, calculate_snr, select_conv_path
        ue, NE = self.ue, self.NE
        H, W, Cin, Cout, k, s, d, pad = sp.H, sp.W, sp.Cin, sp.Cout, sp.k, sp.s, sp.d, sp.pad
        Hout, Wout = sp.Hout, sp.Wout
        K = k * k * Cin
        nominal = 2.0 * Hout * Wout * K * Cout
        pick = select_conv_path(kernel_h=k, kernel_w=k, stride_s=s, dilation=d, c_in=Cin, oc_count=Cout,
                                data_type=TYPE.FLOAT16)
        print(f"== {sp.describe()} -> {Hout}x{Wout}: select_conv_path -> {pick!r} "
              f"(CONV3X3_BF16_FALLBACK={user_dma_core.CONV3X3_BF16_FALLBACK!r})", flush=True)
        torch.manual_seed(0)
        x = torch.relu(torch.randn(Cin, H, W))
        g = torch.randn(Cout, Cin, k, k) * (1.0 / K ** 0.5)
        b = torch.randn(Cout) * 0.1
        xq, gq, bq = (t.to(torch.bfloat16).float() for t in (x, g, b))
        ref = torch.relu(F.conv2d(xq[None], gq, bq, stride=s, padding=pad, dilation=d))[0]
        halo = sp.halo
        X = Map(H, W, Cin, halo, 0)
        X.addr = self.al(X.npix * X.pb)
        grid = torch.zeros(X.npix, Cin)
        grid[:X.Hp * X.Wp].view(X.Hp, X.Wp, Cin)[halo:halo + H, halo:halo + W] = xq.permute(1, 2, 0)
        put(ue, X.addr, grid)
        yhalo = halo if s == 1 else 0                      # stride 1 keeps the model's halo'd chaining layout
        Ys = {}
        for p in paths:
            m = Map(Hout, Wout, Cout, yhalo, 0)
            m.addr = self.al(m.npix * m.pb)
            put(ue, m.addr, torch.zeros(m.npix * Cout))
            Ys[p] = m

        rows = [split(Hout, i, NE) for i in range(NE)]
        results = []

        # ---- direct: im2col + one matmul per chunk of output rows (one chunk per band unless im2col would
        #      exceed IM2COL_BUDGET)
        def setup_direct():
            Wd = gq.permute(0, 2, 3, 1).reshape(Cout, K)    # [Cout, (ky, kx, cin)]
            row_bytes = (X.Wp if s == 1 else Wout) * K * 2
            P = []
            for i in range(NE):
                oa, ob = rows[i]
                rc = max(1, min(ob - oa, IM2COL_BUDGET // row_bytes))
                st = {"bd": self.pwt(i, bq, "bias"), "wd": self.pwt(i, Wd, "w_direct"), "rc": rc}
                if not (k == 1 and s == 1):
                    st["im"] = self.pal(i, rc * row_bytes, "im2col")
                P.append(st)
            return P

        def direct(e, i, P):
            st = P[i]
            oa, ob = rows[i]
            if ob <= oa:
                return 0
            Yd = Ys["direct"]
            gM = e.alloc_isa_reg()
            for c0 in range(oa, ob, st["rc"]):
                n = min(st["rc"], ob - c0)
                if s == 1:
                    # one pass over n padded rows: outputs land on the halo'd map, halo columns re-zeroed below
                    Mr = n * X.Wp
                    if k == 1:
                        a_addr = X.pix(c0, 0)
                    else:
                        a_addr = st["im"]
                        for ky in range(k):
                            if d == 1:
                                scopy(e, X.pix(c0 + ky - pad, -pad), st["im"] + ky * k * Cin * 2, Mr, k * X.pb,
                                      src_jump=X.pb, dst_jump=K * 2)
                            else:
                                for kx in range(k):
                                    scopy(e, X.pix(c0 + ky * d - pad, kx * d - pad),
                                          st["im"] + (ky * k + kx) * Cin * 2, Mr, X.pb, dst_jump=K * 2)
                else:
                    Mr = n * Wout
                    a_addr = st["im"]
                    for r in range(n):
                        oy = c0 + r
                        dst = st["im"] + r * Wout * K * 2
                        for ky in range(k):
                            if d == 1:
                                scopy(e, X.pix(oy * s + ky - pad, -pad), dst + ky * k * Cin * 2, Wout, k * X.pb,
                                      src_jump=s * X.pb, dst_jump=K * 2)
                            else:
                                for kx in range(k):
                                    scopy(e, X.pix(oy * s + ky * d - pad, kx * d - pad),
                                          dst + (ky * k + kx) * Cin * 2, Wout, X.pb, src_jump=s * X.pb,
                                          dst_jump=K * 2)
                e.generate_instruction_add_set(gM, Mr)
                e.matmat_mul_core(M=Mr, K=K, N=Cout, A_DRAM_ADDR=a_addr, B_DRAM_ADDR=st["wd"],
                                  OUTPUT_DRAM_ADDR=Yd.pix(c0, 0), C_DRAM_ADDR=st["bd"], bias_mode="broadcast_N",
                                  gpr_M_reg=gM, clamp_enable=True, clamp_min=0.0, clamp_max=float("inf"))
            n = ob - oa
            if Yd.halo:                                        # halo re-zero (own rows)
                chunk, jump = 2 * Yd.halo * Yd.pb, Yd.Wp * Yd.pb
                e.accelerator_memory_to_sram(self.ZEROS, 0, n * chunk // 2)
                e.sram_to_accelerator_memory(0, Yd.pix(oa, Wout), n * chunk // 2, stride_bytes_per_chunk=chunk,
                                             stride_jump_bytes=jump)
            return 1

        # ---- winograd F(2x2,3x3); dilation 2 through parity sub-grids, 5x5 / 7x7 through 3x3 kernel pieces
        G = torch.tensor([[1., 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])
        mode = sp.winograd_mode
        gs = 2 if mode == "subgrid" else 1                  # sub-grid stride (dilation 2: four parity grids)
        subgrids = [(a, b) for a in range(gs) for b in range(gs)]
        Hs, Ws = Hout // gs, Wout // gs                     # one sub-grid's output map
        TW = Ws // 2
        if mode == "split":                                 # kernel padded to n*3 x n*3, n*n 3x3 pieces
            n3 = -(-k // 3)
            pieces = [(3 * a - pad + 1, 3 * b - pad + 1, a, b) for a in range(n3) for b in range(n3)]
        else:
            pieces = [(0, 0, 0, 0)]

        def setup_winograd():
            assert mode is not None
            twc = wino_chunk(Ws, Cin, Cout)
            assert twc is not None, "one tile column does not fit the SRAM staging"
            gk = gq
            if mode == "split":
                gk = torch.zeros(Cout, Cin, 3 * n3, 3 * n3)
                gk[:, :, :k, :k] = gq
            sgn = lambda r, c: (-1.0 if r == 2 else 1.0) * (-1.0 if c == 2 else 1.0)
            P = []
            for i in range(NE):
                ta, tb = split(Hs // 2, i, NE)
                tg = max(1, min(tb - ta, WINO_BUDGET // (16 * TW * (Cin + Cout) * 2)))   # tile rows per group
                nt = tg * TW
                us = []
                for sy, sx, a, b in pieces:
                    U = torch.einsum("ik,ockl,jl->ocij", G, gk[:, :, 3 * a:3 * a + 3, 3 * b:3 * b + 3], G)
                    us.append([self.pwt(i, (sgn(r, c) * U[:, :, r, c]).contiguous(), f"us{a}{b}_{r}{c}")
                               for r in range(4) for c in range(4)])
                st = {"tiles": (ta, tb), "tg": tg, "twc": twc, "bd": self.pwt(i, bq, "bias"), "us": us}
                st["D"] = [self.pal(i, nt * Cin * 2, f"D{p}") for p in range(16)]
                st["M"] = [self.pal(i, nt * Cout * 2, f"M{p}") for p in range(16)]
                P.append(st)
            return P

        def winograd(e, i, P):
            """winograd_probe 'weo': even/odd input stage, transforms in SRAM, 16 matmuls, output transform in SRAM.
            Per sub-grid and tile-row group; kernel pieces accumulate into the same 16 M planes."""
            st = P[i]
            ta, tb = st["tiles"]
            if tb <= ta:
                return 0
            Yw = Ys["winograd"]
            twc = st["twc"]
            gM = e.alloc_isa_reg()
            D, Mx = st["D"], st["M"]
            A0, B0 = 0x00000, SRAM_BANK
            add = lambda a, b_, o, n: e.eltwise_add_core(a, b_, o, n)
            sub = lambda a, b_, o, n: e.eltwise_sub_core(a, b_, o, n)
            sz = up128((twc + 1) * X.pb)
            DS = lambda r, par: (A0 + (r * 2 + par) * sz) if r < 2 else (B0 + ((r - 2) * 2 + par) * sz)
            TA = lambda k_, par: A0 + (4 + k_ * 2 + par) * sz
            TB = lambda k_, par: B0 + (4 + k_ * 2 + par) * sz
            Pout = up128(twc * Cout * 2)
            Msr = lambda r, c: A0 + c * Pout if r == 1 else B0 + ({0: 0, 2: 1, 3: 2}[r] * 4 + c) * Pout
            Ssr = lambda i_, c: A0 + (4 + i_) * Pout if c == 1 else B0 + (12 + i_ * 3 + {0: 0, 2: 1, 3: 2}[c]) * Pout
            T0, T1 = A0 + 6 * Pout, A0 + 7 * Pout
            Ysr = lambda k_: A0 + (8 + k_) * Pout
            chunks = [(c0, min(twc, TW - c0)) for c0 in range(0, TW, twc)]
            for ga, gb in subgrids:
                xin = lambda yy, xx: X.pix(gs * yy + ga, gs * xx + gb)       # sub-grid coords -> map pixel
                for g0 in range(ta, tb, st["tg"]):                          # tile-row groups bound the D / M planes
                    g1 = min(g0 + st["tg"], tb)
                    nt = (g1 - g0) * TW
                    for pi, (sy, sx, _a, _b) in enumerate(pieces):
                        for ty in range(g0, g1):
                            for c0, n in chunks:
                                off = (ty - g0) * TW + c0
                                ne, nel = (n + 1) * Cin, n * Cin
                                for r in range(4):                          # each source row once: even / odd pixels
                                    for par in (0, 1):
                                        e.accelerator_memory_to_sram(xin(2 * ty - 1 + r + sy, 2 * c0 - 1 + par + sx),
                                                                     DS(r, par), ne, stride_bytes_per_chunk=X.pb,
                                                                     stride_jump_bytes=2 * gs * X.pb)
                                for T in (TA, TB):                          # rows: T0=D0-D2, T1=D1+D2, -T2=D1-D2, T3=D1-D3
                                    for par in (0, 1):
                                        sub(DS(0, par), DS(2, par), T(0, par), ne)
                                        add(DS(1, par), DS(2, par), T(1, par), ne)
                                        sub(DS(1, par), DS(2, par), T(2, par), ne)
                                        sub(DS(1, par), DS(3, par), T(3, par), ne)
                                for k_ in range(4):                         # cols on tile windows (contiguous slices)
                                    Vs = [A0 + q * sz for q in range(4)]
                                    sub(TA(k_, 0), TB(k_, 0) + X.pb, Vs[0], nel)
                                    add(TA(k_, 1), TB(k_, 0) + X.pb, Vs[1], nel)
                                    sub(TA(k_, 1), TB(k_, 0) + X.pb, Vs[2], nel)
                                    sub(TA(k_, 1), TB(k_, 1) + X.pb, Vs[3], nel)
                                    for q in range(4):
                                        e.sram_to_accelerator_memory(Vs[q], D[k_ * 4 + q] + off * Cin * 2, nel)
                        for xi in range(16):                                # 16 matmuls; pieces after the first accumulate
                            e.generate_instruction_add_set(gM, nt)
                            if pi == 0:
                                kw = dict(C_DRAM_ADDR=st["bd"], bias_mode="broadcast_N") if xi == 5 else {}
                            else:
                                kw = dict(C_DRAM_ADDR=Mx[xi], bias_mode="full_matrix")
                            e.matmat_mul_core(M=nt, K=Cin, N=Cout, A_DRAM_ADDR=D[xi], B_DRAM_ADDR=st["us"][pi][xi],
                                              OUTPUT_DRAM_ADDR=Mx[xi], gpr_M_reg=gM, **kw)
                    Y = Mx[:4]
                    for ty in range(g0, g1):
                        for c0, n in chunks:
                            off = (ty - g0) * TW + c0
                            nel = n * Cout
                            for r in range(4):
                                for c in range(4):
                                    e.accelerator_memory_to_sram(Mx[r * 4 + c] + off * Cout * 2, Msr(r, c), nel)
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
                            for k_ in range(4):
                                e.sram_to_accelerator_memory(Ysr(k_), Y[k_] + off * Cout * 2, nel)
                    ro = nt * Cout // 64
                    for yp in Y:                                            # ReLU (identity-matmul clamp)
                        e.generate_instruction_add_set(gM, ro)
                        e.activation_core(M=ro, N=64, A_DRAM_ADDR=yp, OUTPUT_DRAM_ADDR=yp,
                                          IDENTITY_DRAM_ADDR=self.I64, activation="clamp", gpr_M_reg=gM)
                    for i_ in range(2):
                        for j in range(2):
                            for ty in range(g0, g1):
                                scopy(e, Y[2 * i_ + j] + (ty - g0) * TW * Cout * 2,
                                      Yw.pix(gs * (2 * ty + i_) + ga, gs * j + gb), TW, Cout * 2,
                                      dst_jump=2 * gs * Yw.pb)
            return 16 * len(pieces) * len(subgrids)

        # ---- native CONV2D walker (FLOAT16 or IF8 weights, weight_reuse)
        quant = {}

        def setup_native(fmt):
            dtype, limit, bpb = ((TYPE.FLOAT16, 4096, 128) if fmt == "native16" else (TYPE.IF8, 8192, 64))
            if dtype == TYPE.IF8 and "codes" not in quant:
                quant["codes"], quant["scales"], deq = quantize_if8_channels(gq)
                quant["ref"] = torch.relu(F.conv2d(xq[None], deq, bq, stride=s, padding=pad, dilation=d))[0]
            per_rows = {}
            for i in range(NE):
                oa, ob = rows[i]
                n = ob - oa
                if n <= 0 or n in per_rows:
                    continue
                out_h, out_w, occ, tiles = user_dma_core.plan_conv2d_layer_tiles(
                    c_in=Cin, oc_count=Cout, in_h=n * s, in_w=W, kernel_h=k, kernel_w=k, stride_s=s, pad=pad,
                    dilation=d, bias_enabled=True, weight_block_limit=limit, act_uram_addr=0,
                    wb_uram_addr=ACT_URAM_LINES)
                assert (out_h, out_w) == (n, Wout), (out_h, out_w, n, Wout)
                assert occ * k * k * (Cin // V) * bpb <= URAM_B_BYTES, (fmt, occ)
                per_rows[n] = (occ, tiles, user_dma_core.conv2d_tile_geometry_groups(tiles))
            occ = next(iter(per_rows.values()))[0]
            assert all(v[0] == occ for v in per_rows.values()), (fmt, per_rows)
            chunks = []
            for oc0 in range(0, Cout, occ):
                sl = slice(oc0, oc0 + occ)
                if dtype == TYPE.FLOAT16:
                    wblk = user_dma_core.conv2d_pack_weight_stream(gq[sl], 1, 1, TYPE.FLOAT16)
                    sblk = None
                else:
                    wblk = user_dma_core.conv2d_pack_weight_stream(quant["codes"][sl], 1, 1, TYPE.IF8)
                    sblk = user_dma_core.conv2d_pack_scale_stream(quant["scales"][sl], occ, k * k * (Cin // V), 1, 1)
                bblk = user_dma_core.conv2d_pack_bias_stream(bq[sl].to(torch.bfloat16), 1, 1)
                chunks.append((oc0, wblk.numpy().tobytes(),
                               None if sblk is None else user_dma_core.tensor_le_bytes(sblk.contiguous()),
                               user_dma_core.tensor_le_bytes(bblk)))
            print(f"   {fmt}: oc chunk {occ} ({Cout // occ} chunks), pattern "
                  f"{occ * k * k * (Cin // V) * bpb // 1024} KB, tiles per band {[len(v[1]) for v in per_rows.values()]}",
                  flush=True)
            tile_px = max(int(t[2]) * int(t[3]) for v in per_rows.values() for t in v[1])
            P = []
            for i in range(NE):
                P.append({"dtype": dtype, "occ": occ, "per_rows": per_rows,
                          "w": [(oc0, self.pwb(i, wb, f"{fmt}_w{oc0}"),
                                 None if sb is None else self.pwb(i, sb, f"{fmt}_s{oc0}"),
                                 self.pwb(i, bb, f"{fmt}_b{oc0}")) for oc0, wb, sb, bb in chunks],
                          "wbs": self.pal(i, tile_px * occ * 2, f"{fmt}_wb_bounce")})
            return P

        def native(e, i, P, fmt):
            """CONV2D with weight_reuse over this engine's row band: window from the halo map -> URAM A,
            one launch per (oc chunk, tile), writeback scattered into the destination map."""
            st = P[i]
            oa, ob = rows[i]
            if ob <= oa:
                return 0
            Yn = Ys[fmt]
            dtype, occ = st["dtype"], st["occ"]
            _, tiles, groups = st["per_rows"][ob - oa]
            ct = Cin // V
            taps = k * k * ct
            mode, la, lb = self.lalu_relu
            launches = 0
            for oc0, w_addr, s_addr, b_addr in st["w"]:
                for start, stop, th, tw, win_h, win_w in groups:
                    if s_addr is not None:
                        e.accelerator_memory_to_scale_sram(s_addr, occ * taps)
                    e.accelerator_memory_to_bias_sram(b_addr, occ)
                    for tile in tiles[start:stop]:
                        oy, ox, th_, tw_, y0, x0, wh, ww = (int(v) for v in tile)
                        a = X.pix(oa * s - pad + y0, x0 - pad)        # padded-window origin in the halo map
                        row = ww * X.pb
                        if row == X.pitch:
                            e.accelerator_memory_to_sram(a, 0, wh * row // 2)
                        else:
                            e.accelerator_memory_to_sram(a, 0, wh * row // 2, stride_bytes_per_chunk=row,
                                                         stride_jump_bytes=X.pitch)
                        e.start_queue_for_conv2d_operation(
                            act_sram_start_addr=0, output_sram_wb_addr=WB_SRAM, weights_dram_addr=w_addr,
                            kernel_w=k, kernel_h=k, ct=ct, oc_count=occ, out_w=tw_, out_h=th_, w_pad=ww,
                            stride_s=s, data_type=dtype, bias_enable=True, lalu_mode=mode, lalu_a=la, lalu_b=lb,
                            dilation=d, gather=False, c_in=Cin, weight_reuse=True)
                        launches += 1
                        oc_b = occ * 2
                        if oc_b == Yn.pb:                                 # whole pixels per row
                            row = tw_ * Yn.pb
                            if th_ == 1 or row == Yn.pitch:
                                e.sram_to_accelerator_memory(WB_SRAM, Yn.pix(oa + oy, ox), th_ * row // 2)
                            else:
                                e.sram_to_accelerator_memory(WB_SRAM, Yn.pix(oa + oy, ox), th_ * row // 2,
                                                             stride_bytes_per_chunk=row, stride_jump_bytes=Yn.pitch)
                        elif th_ == 1 or (tw_ * oc_b) % 128 == 0:         # oc chunk at its lane offset, per row
                            for r in range(th_):
                                e.sram_to_accelerator_memory(WB_SRAM + r * tw_ * oc_b, Yn.pix(oa + oy + r, ox) + oc0 * 2,
                                                             tw_ * oc_b // 2, stride_bytes_per_chunk=oc_b,
                                                             stride_jump_bytes=Yn.pb)
                        else:                                             # rows not on 128 B SRAM lines: bounce via DRAM
                            bounce = st["wbs"]
                            e.sram_to_accelerator_memory(WB_SRAM, bounce, up128(th_ * tw_ * oc_b) // 2)
                            for r in range(th_):
                                e.accelerator_memory_to_sram(bounce + r * tw_ * oc_b, 0, tw_ * oc_b // 2)
                                e.sram_to_accelerator_memory(0, Yn.pix(oa + oy + r, ox) + oc0 * 2,
                                                             tw_ * oc_b // 2, stride_bytes_per_chunk=oc_b,
                                                             stride_jump_bytes=Yn.pb)
            return launches

        setups = {"direct": setup_direct, "winograd": setup_winograd,
                  "native16": lambda: setup_native("native16"), "native8": lambda: setup_native("native8")}
        bodies = {"direct": direct, "winograd": winograd,
                  "native16": lambda e, i, P: native(e, i, P, "native16"),
                  "native8": lambda e, i, P: native(e, i, P, "native8")}

        def record(meth, status, **kw):
            rec = {"case": sp.name, "group": sp.group, "kind": sp.kind, "wmode": sp.winograd_mode,
                   "H": H, "W": W, "Cin": Cin, "Cout": Cout,
                   "k": k, "s": s, "d": d, "engines": NE, "path": meth, "status": status, "pick": pick}
            rec.update(kw)
            results.append(rec)
            return rec

        if auto is not None:
            # UnifiedEngine.conv2d_auto on one engine: the wrapper picks the path, we log which emitter actually
            # ran (walker = its own conv2d_core call; winograd / matmul / im2col = the emitters above), then check
            # the route against the expected one and the result against torch.
            assert NE == 1, "--auto runs the wrapper on one engine (its walker path is one whole-conv launch)"
            dtype = TYPE.IF8 if auto.get("dtype") == "if8" else TYPE.FLOAT16
            ref_a = ref
            wsrc = gq
            ct = (Cin + V - 1) // V
            s_addr = 0
            if dtype == TYPE.IF8:
                wsrc, scales, deq = quantize_if8_channels(gq)
                ref_a = torch.relu(F.conv2d(xq[None], deq, bq, stride=s, padding=pad, dilation=d))[0]
                sblk = user_dma_core.conv2d_pack_scale_stream(scales, Cout, k * k * ct, 1, 1)
                sb = user_dma_core.tensor_le_bytes(sblk.contiguous())
                s_addr = self.al(len(sb))
                put_bytes(ue, s_addr, sb)
            amap = user_dma_core.conv2d_pack_activation_map(xq, pad)
            act = self.al(amap.numel() * 2)
            put(ue, act, amap)
            wb = user_dma_core.conv2d_pack_weight_stream(wsrc, 1, 1, dtype).numpy().tobytes()
            w_addr = self.al(len(wb))
            put_bytes(ue, w_addr, wb)
            bb = user_dma_core.tensor_le_bytes(user_dma_core.conv2d_pack_bias_stream(bq.to(torch.bfloat16), 1, 1))
            b_addr = self.al(len(bb))
            put_bytes(ue, b_addr, bb)
            out = self.al(Hout * Wout * Cout * 2)
            self.restore_arena()
            Pw = setups["winograd"]() if sp.winograd_applies else None
            Pd = setups["direct"]()
            ran = []

            def body(e, i):
                core = e.conv2d_core

                def logged(**kw):
                    ran.append("walker")
                    return core(**kw)
                e.conv2d_core = logged                          # instance attribute shadows the method
                try:
                    e.conv2d_auto(
                        winograd=lambda: (ran.append("winograd"), winograd(e, i, Pw))[1],
                        matmul=lambda: (ran.append("matmul"), direct(e, i, Pd))[1],
                        im2col=lambda: (ran.append("im2col"), direct(e, i, Pd))[1],
                        allow_winograd=auto.get("allow_winograd", True),
                        ACT_DRAM_ADDR=act, WEIGHTS_DRAM_ADDR=w_addr, SCALE_DRAM_ADDR=s_addr, OUTPUT_DRAM_ADDR=out,
                        c_in=Cin, in_h=H, in_w=W, kernel_h=k, kernel_w=k, stride_s=s, pad=pad, dilation=d,
                        oc_count=Cout, data_type=dtype, BIAS_DRAM_ADDR=b_addr, relu_enable=True)
                finally:
                    del e.conv2d_core
                return 1

            built = self.run(body, runs=1, reps=1)
            expect = auto["expect"]
            if isinstance(built, Exception):
                record("auto", "fail", expect=expect, route=",".join(ran), reason=f"compile: {built}"[:200])
                print(f"  {sp.name:22s} FAIL  expected {expect}, wrapper ran {ran or 'nothing'} -- build error: "
                      f"{str(built)[:160]}", flush=True)
                return results
            ms = built[0]
            route = ran[0] if len(ran) == 1 else ",".join(ran)
            if route == "walker":
                hw = get(ue, out, Hout * Wout * Cout).view(Hout, Wout, Cout).permute(2, 0, 1)
            else:
                ym = Ys["winograd" if route == "winograd" else "direct"]
                hw = get(ue, ym.addr, ym.Hp * ym.Wp * Cout).view(ym.Hp, ym.Wp, Cout)
                hw = hw[ym.halo:ym.halo + Hout, ym.halo:ym.halo + Wout].permute(2, 0, 1)
            snr = float(calculate_snr(ref_a.flatten(), hw.flatten()))
            bad = int((~torch.isfinite(hw)).sum())
            good = route == expect and snr >= 30 and bad == 0
            record("auto", "ok" if good else "fail", expect=expect, route=route, snr=snr, ms=ms,
                   gflops=nominal / (ms * 1e-3) / 1e9)
            print(f"  {sp.name:22s} {'PASS' if good else 'FAIL'}  expected {expect:8s} wrapper ran {route:8s} "
                  f"SNR {snr:6.2f} dB  non-finite {bad}  {ms:.3f} ms", flush=True)
            ue.reset_tensor_dram_addr()
            self.al(64 * 64 * 2 + STAGE)
            return results

        res = {}
        for meth in paths:
            if meth == "winograd" and not sp.winograd_applies:
                record(meth, "n/a")
                print(f"  {sp.name:22s} {meth:9s} n/a (stride {s}: no F(2x2,3x3) piece in its polyphase split)")
                continue
            self.restore_arena()
            try:
                P = setups[meth]()
            except (AssertionError, MemoryError, ValueError) as ex:
                record(meth, "skip", reason=str(ex)[:200])
                print(f"  {sp.name:22s} {meth:9s} skip: {str(ex)[:160]}", flush=True)
                continue
            print(f"  {sp.name:22s} {meth:9s} compiling + running ...", flush=True)
            built = self.run(lambda e, i: bodies[meth](e, i, P))
            if isinstance(built, Exception):
                record(meth, "skip", reason=f"compile: {built}"[:200])
                print(f"  {sp.name:22s} {meth:9s} skip (compile): {str(built)[:160]}", flush=True)
                continue
            ms, ninst, nl = built
            ym = Ys[meth]
            hw = get(ue, ym.addr, ym.Hp * ym.Wp * Cout).view(ym.Hp, ym.Wp, Cout)
            hw = hw[ym.halo:ym.halo + Hout, ym.halo:ym.halo + Wout].permute(2, 0, 1)
            snr = float(calculate_snr(ref.flatten(), hw.flatten()))
            extra = ""
            if meth == "native8":
                snr8 = float(calculate_snr(quant["ref"].flatten(), hw.flatten()))
                extra = f"  (vs dequant IF8 ref {snr8:6.2f} dB)"
                good_snr = snr8 >= 30
            else:
                good_snr = snr >= 30
            bad = int((~torch.isfinite(hw)).sum())
            good = good_snr and bad == 0
            gf = nominal / (ms * 1e-3) / 1e9
            res[meth] = ms
            record(meth, "ok" if good else "fail", ms=ms, gflops=gf, snr=snr, launches=nl, instr=ninst)
            print(f"  {sp.name:22s} {Hout}x{Wout} {Cin}->{Cout}  {meth:9s} {ms:8.3f} ms/conv  {gf:7.1f} GFLOPS nominal "
                  f"({100 * gf / self.peak:5.1f}% of {self.peak:.0f} peak)  {nl} launches/eng  {ninst} primary instr  "
                  f"SNR {snr:6.2f} dB{extra}  non-finite {bad}  {'PASS' if good else 'FAIL'}", flush=True)
        base = "native16" if "native16" in res else ("direct" if "direct" in res else None)
        if base and len(res) > 1:
            order = sorted(res, key=res.get)
            print(f"  {sp.name:22s} order: " + " < ".join(f"{m} {res[m]:.3f}" for m in order) + "   |  " +
                  "  ".join(f"{m} {res[base] / res[m]:.2f}x vs {base}" for m in order if m != base), flush=True)
        ue.reset_tensor_dram_addr()
        self.al(64 * 64 * 2 + STAGE)                       # keep I64 / ZEROS at the same addresses
        return results

    def run(self, body, runs=3, reps=None):
        """Capture one program (reps x body on every engine), run it `runs` times; returns (ms/conv, instr,
        launches) or the build exception (nothing has touched the board at that point)."""
        reps = self.args.reps if reps is None else reps
        ue, sched, NE = self.ue, self.sched, self.NE
        ue.reset_program_dram_addr()
        for w in sched.workers:
            w.reset_program_dram_addr()
        ue.start_capture()
        ue.reset_isa_reg_counter()
        ue.reset_inst_ptr_counter()
        launches = [0] * NE
        try:
            sched.begin_program()
            for _ in range(reps):
                def _b(ctx):
                    launches[ctx.engine_idx] = body(ctx.unsafe_ue, ctx.engine_idx)
                sched.sharded_region(64 * NE, _b)
            progs = sched.finalize()
        except (AssertionError, MemoryError, ValueError) as ex:
            sched.abort_program()
            ue.stop_capture()
            ue.clear_capture_buffer()
            return ex
        ue.stop_capture()
        ue.generate_instruction_halt()
        addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        ninst = ue.capture_count
        ue.clear_capture_buffer()
        ts = []
        for j in range(runs):
            t0 = time.perf_counter()
            sched.start_workers(progs)
            ue.start_execute_from_dram(addr)
            ue.wait_queue(120.0)
            ts.append((time.perf_counter() - t0) * 1000)
            if j + 1 < runs:
                time.sleep(0.5)
        return sorted(ts)[len(ts) // 2] / reps, ninst, max(launches)


# ----------------------------------------------------------------------------------------------- reporting
def best_other(rows_by_path):
    """Fastest measured non-Winograd bf16 path: (name, ms) or None."""
    cand = [(m, r["ms"]) for m, r in rows_by_path.items() if m in ("native16", "direct") and r["status"] == "ok"]
    return min(cand, key=lambda t: t[1]) if cand else None


def by_case(results):
    out = {}
    for r in results:
        out.setdefault(r["case"], {})[r["path"]] = r
    return out


def print_summary(results, peak=None):
    ok = all(r["status"] in ("ok", "n/a", "skip") for r in results)
    print()
    print(f"{'case':22s} {'kind':8s} {'path':9s} {'ms/conv':>9s} {'GFLOPS':>8s} {'%peak':>6s} {'SNR':>7s}")
    for r in results:
        if r["status"] in ("ok", "fail"):
            pk = f"{100 * r['gflops'] / peak:6.1f}" if peak else "     -"
            print(f"{r['case']:22s} {r['kind']:8s} {r['path']:9s} {r['ms']:9.3f} {r['gflops']:8.1f} {pk} "
                  f"{r['snr']:7.2f}")
        else:
            print(f"{r['case']:22s} {r['kind']:8s} {r['path']:9s} {r['status']:>9s}")
    cases = by_case(results)
    if any({"direct", "native16"} <= set(c) for c in cases.values()):
        print("\nbf16 fallback verdict per case (native16 faster than direct?):")
        for name, c in cases.items():
            if all(c.get(m, {}).get("status") == "ok" for m in ("direct", "native16")):
                n16, dr = c["native16"]["ms"], c["direct"]["ms"]
                print(f"  {name:22s} native16 {n16:.3f} ms vs direct {dr:.3f} ms -> {'walker' if n16 < dr else 'im2col'}")
        print("set user_dma_core.CONV3X3_BF16_FALLBACK to the winner.")
    wins = []
    for name, c in cases.items():
        w = c.get("winograd")
        bo = best_other(c)
        if w and w["status"] == "ok" and bo:
            wins.append((name, w["kind"], bo[1] / w["ms"], bo[0], w["pick"]))
    if wins:
        print("\nWinograd vs the best other bf16 path (>1 = Winograd faster), and what select_conv_path picks:")
        for name, kind, sp_, other, pick in wins:
            tag = "WIN " if sp_ > 1.05 else ("tie " if sp_ >= 0.95 else "LOSS")
            print(f"  {tag} {name:22s} {kind:6s} {sp_:5.2f}x vs {other:8s} selector -> {pick}")
    print("CONV3X3_PATHS_PROBE:", "PASS" if ok else "FAIL")
    return ok


INK, INK2, MUTED, SURFACE, GRID_LINE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SERIES = {"winograd": "#2a78d6", "native16": "#eb6834", "direct": "#1baf7a", "native8": "#eda100"}
LABEL = {"winograd": "Winograd F(2x2,3x3) (direct / sub-grid / kernel split)", "native16": "native CONV2D, bf16 (baseline = 1.0)",
         "direct": "im2col + matmul", "native8": "native CONV2D, IF8"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    return plt


def _measured(r):
    return r is not None and r["status"] in ("ok", "fail")


def _shape_label(r):
    shape = f"{r['H']}x{r['W']} · {r['Cin']}→{r['Cout']}"
    return f"{r['case']}  {shape}" if r["group"] == "case" else shape


def plot(results, png, title):
    """Speedup vs native16 for every path: one panel per conv type, one bar group per (map size, channels)."""
    plt = _mpl()
    from matplotlib.patches import Patch
    from user_dma_core import CONV3X3_WINOGRAD_MIN_CIN as thr
    cases = by_case(results)
    kinds = []
    for c in cases.values():
        kd = next(iter(c.values()))["kind"]
        if kd not in kinds:
            kinds.append(kd)
    order = [lbl.replace("x", "x") for lbl, *_ in CONV_TYPES]
    kinds.sort(key=lambda kd: order.index(kd) if kd in order else len(order))
    paths = [p for p in PATH_ORDER if any(_measured(c.get(p)) for c in cases.values())]
    xmax = max([1.0] + [c["native16"]["ms"] / r["ms"] for c in cases.values() if _measured(c.get("native16"))
                        for r in c.values() if _measured(r)])
    ncol = min(3, len(kinds))
    nrow = (len(kinds) + ncol - 1) // ncol
    per_panel = max(sum(1 for c in cases.values() if next(iter(c.values()))["kind"] == kd) for kd in kinds)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, 1.2 + 0.36 * per_panel * nrow + 0.9 * nrow),
                             squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    for ax in axes.flat[len(kinds):]:
        ax.set_visible(False)
    for ax, kd in zip(axes.flat, kinds):
        rows_ = sorted((c for c in cases.values() if next(iter(c.values()))["kind"] == kd),
                       key=lambda c: (next(iter(c.values()))["H"] * next(iter(c.values()))["W"],
                                      next(iter(c.values()))["Cin"]))
        wino_applies = any(c.get("winograd", {}).get("status") != "n/a" for c in rows_ if "winograd" in c)
        here = [p for p in paths if p != "winograd" or wino_applies]
        bar_h = 0.84 / max(1, len(here))
        prev_hw = None
        for yi, c in enumerate(rows_):
            r0 = next(iter(c.values()))
            if prev_hw is not None and (r0["H"], r0["W"]) != prev_hw:
                ax.axhline(yi - 0.5, color=GRID_LINE, lw=1.2)
            prev_hw = (r0["H"], r0["W"])
            base = c.get("native16")
            for pi, p in enumerate(here):
                y = yi - 0.42 + bar_h * (pi + 0.5)
                r = c.get(p)
                if _measured(r) and _measured(base):
                    v = base["ms"] / r["ms"]
                    ax.barh(y, v, height=bar_h * 0.8, color=SERIES[p], lw=0)
                    if p != "native16":                          # baseline bar is 1.00 by definition
                        ax.text(v + xmax * 0.012, y, f"{v:.2f}x", va="center", fontsize=6.8, color=INK2)
                elif r is not None and r["status"] != "n/a":
                    ax.text(xmax * 0.012, y, f"{p}: {r['status']}", va="center", fontsize=6.8, color=MUTED,
                            style="italic")
        ax.axvline(1.0, color=INK2, lw=1)
        ax.set_yticks(range(len(rows_)), [_shape_label(next(iter(c.values()))) for c in rows_])
        ax.set_ylim(len(rows_) - 0.5, -0.5)
        ax.set_xlim(0, xmax * 1.15)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID_LINE, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("speedup vs native CONV2D bf16 (higher is faster)")
        note = (f"Winograd applies; select_conv_path uses it from Cin ≥ {thr}" if wino_applies
                else "Winograd n/a: needs 3x3, stride 1, dilation 1")
        ax.set_title(f"{kd} conv\n{note}", fontsize=10.5)
        ax.title.set_linespacing(1.5)
    fig.legend(handles=[Patch(color=SERIES[p], label=LABEL[p]) for p in paths], loc="upper right",
               bbox_to_anchor=(0.995, 0.997), ncol=len(paths), frameon=False, fontsize=9.5)
    fig.text(0.01, 0.003, "rows: input map H x W · input channels → output channels; "
             "stride-2 convs produce a half-size map", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"graph: {png}")


def plot_snr(results, png, title):
    """Speed vs accuracy for every measured conv."""
    plt = _mpl()
    from matplotlib.ticker import FuncFormatter
    cases = by_case(results)
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for p in PATH_ORDER:
        xs, ys = [], []
        for c in cases.values():
            r, base = c.get(p), c.get("native16")
            if _measured(r) and _measured(base):
                xs.append(base["ms"] / r["ms"])
                ys.append(r["snr"])
        if xs:
            ax.scatter(xs, ys, s=42, color=SERIES[p], edgecolor=SURFACE, linewidth=1.5, label=LABEL[p], zorder=4 if p == "native16" else 3)
    ax.axhline(30, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.99, 30.4, "30 dB pass line", fontsize=7.5, color=INK2, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())
    ax.axvline(1.0, color=GRID_LINE, lw=1)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_xlabel("speedup vs native CONV2D bf16 (log scale)")
    ax.set_ylabel("SNR vs fp32 reference (dB)")
    ax.set_title(f"Speed vs accuracy, every measured conv\n{title}", fontsize=10)
    ax.grid(color=GRID_LINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(min(lo_y, 28), max(hi_y, 32))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=False, fontsize=8)
    else:                                                    # e.g. --paths winograd: no native16 baseline
        ax.text(0.5, 0.5, "no native16 run to compare against", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"graph: {png}")


def _type_rows(cases, kd):
    return sorted((c for c in cases.values() if next(iter(c.values()))["kind"] == kd),
                  key=lambda c: (next(iter(c.values()))["H"] * next(iter(c.values()))["W"],
                                 next(iter(c.values()))["Cin"]))


def _wino_note(rows_, thr):
    modes = {r.get("wmode") for c in rows_ for r in c.values()} - {None}
    if not modes and not any(c.get("winograd", {}).get("status") not in (None, "n/a") for c in rows_):
        r0 = next(iter(rows_[0].values()))
        if r0["k"] == 1:
            return "Winograd n/a: a 1x1 conv shares no inputs between outputs (it is a plain matmul)"
        if r0["s"] > 1:
            return f"Winograd n/a: stride {r0['s']} (no F(2x2,3x3) piece in its polyphase split)"
        return "Winograd not run in this sweep"
    mode = modes.pop() if modes else "plain"
    r0 = next(iter(rows_[0].values()))
    note = WINO_NOTE.get(mode, "Winograd").format(n=(-(-r0["k"] // 3)) ** 2)
    return note + (f"; select_conv_path uses it from Cin ≥ {thr}" if mode == "plain" else "")


def plot_split(results, prefix, title):
    """One readable PNG per conv type: <prefix>_<type>.png, rows = map size · channels, one bar per path."""
    plt = _mpl()
    from matplotlib.patches import Patch
    from user_dma_core import CONV3X3_WINOGRAD_MIN_CIN as thr
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    cases = by_case(results)
    kinds = []
    for c in cases.values():
        kd = next(iter(c.values()))["kind"]
        if kd not in kinds:
            kinds.append(kd)
    order = [lbl for lbl, *_ in CONV_TYPES]
    kinds.sort(key=lambda kd: order.index(kd) if kd in order else len(order))
    out = []
    for kd in kinds:
        rows_ = _type_rows(cases, kd)
        paths = [p for p in PATH_ORDER if any(_measured(c.get(p)) for c in rows_)]
        if not paths:
            continue
        xmax = max([1.0] + [c["native16"]["ms"] / r["ms"] for c in rows_ if _measured(c.get("native16"))
                            for r in c.values() if _measured(r)])
        bar_h = 0.82 / len(paths)
        fig, ax = plt.subplots(figsize=(9.5, 1.9 + 0.17 * len(paths) * len(rows_) + 0.12 * len(rows_)))
        prev_hw = None
        for yi, c in enumerate(rows_):
            r0 = next(iter(c.values()))
            if prev_hw is not None and (r0["H"], r0["W"]) != prev_hw:
                ax.axhline(yi - 0.5, color=GRID_LINE, lw=1.2)
            prev_hw = (r0["H"], r0["W"])
            base = c.get("native16")
            for pi, p in enumerate(paths):
                y = yi - 0.41 + bar_h * (pi + 0.5)
                r = c.get(p)
                if _measured(r) and _measured(base):
                    v = base["ms"] / r["ms"]
                    ax.barh(y, v, height=bar_h * 0.8, color=SERIES[p], lw=0)
                    if p != "native16":
                        ax.text(v + xmax * 0.012, y, f"{v:.2f}x", va="center", fontsize=7.5, color=INK2)
                elif _measured(r):                                   # ran, but native16 could not: no ratio
                    ax.text(xmax * 0.012, y, f"{p}: {r['ms']:.2f} ms (native16 can't run this shape)", va="center",
                            fontsize=7.5, color=MUTED, style="italic")
                elif r is not None and r["status"] != "n/a":
                    ax.text(xmax * 0.012, y, f"{p}: {r['status']}", va="center", fontsize=7.5, color=MUTED,
                            style="italic")
        ax.axvline(1.0, color=INK2, lw=1)
        ax.set_yticks(range(len(rows_)), [_shape_label(next(iter(c.values()))) for c in rows_])
        ax.set_ylim(len(rows_) - 0.5, -0.5)
        ax.set_xlim(0, xmax * 1.14)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID_LINE, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("speedup vs native CONV2D bf16 (higher is faster)")
        ax.set_title(f"{kd} conv\n{_wino_note(rows_, thr)}", fontsize=12)
        ax.title.set_linespacing(1.45)
        fh = fig.get_figheight()
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK2)
        fig.legend(handles=[Patch(color=SERIES[p], label=LABEL[p]) for p in paths], loc="upper left",
                   bbox_to_anchor=(0.005, 1 - 0.40 / fh), ncol=2, frameon=False, fontsize=9)
        fig.text(0.01, 0.005, "rows: input map H x W · input channels → output channels"
                 + ("; stride 2 halves the output map" if "s2" in kd else ""), fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 1 - (0.58 + 0.2 * ((len(paths) - 1) // 2)) / fh))
        png = f"{prefix}_{kd.replace(' ', '_')}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        out.append(png)
        print(f"graph: {png}")
    return out


def plot_all(results, prefix, title):
    plot(results, prefix + ".png", title)
    plot_snr(results, prefix + "_snr.png", title)
    plot_split(results, prefix, title)


# ----------------------------------------------------------------------------------------------- main
def build_specs(args):
    specs = []
    if args.auto:
        for name, k, st, d, cin, cout, dt, allow, expect in AUTO_CASES:
            sp = Spec(name, "auto", 12, 20, cin, cout, k=k, s=st, d=d)
            sp.auto = {"dtype": dt, "allow_winograd": allow, "expect": expect}
            specs.append(sp)
    elif args.sweep:
        spatial = [tuple(int(v) for v in hw.split("x")) for hw in args.spatial.split(",")]
        chans = [int(c) for c in args.channels.split(",")]
        want = set(t.strip() for t in args.types.split(",")) if args.types else None
        for label, k, s, d in CONV_TYPES:
            if want and label not in want:
                continue
            for H, W in spatial:
                for c in chans:
                    specs.append(Spec(f"{label} {H}x{W} c{c}", "sweep", H, W, c, c, k=k, s=s, d=d))
    else:
        for name in args.cases.split(","):
            H, W, Cin, Cout, halo = CASES[name]
            specs.append(Spec(name, "case", H, W, Cin, Cout, halo=halo))
    return specs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--cases", default="dec3,enc1,dec1,dec0")
    ap.add_argument("--paths", default=None,
                    help="default: direct,winograd,native16,native8 (cases) / winograd,native16,direct (sweep)")
    ap.add_argument("--reps", type=int, default=3, help="convs per program (time = median run / reps)")
    ap.add_argument("--sweep", action="store_true", help="every conv type x map size x channels instead of the cases")
    ap.add_argument("--spatial", default=SWEEP_SPATIAL, help="--sweep input map sizes, HxW list")
    ap.add_argument("--channels", default=SWEEP_CHANNELS, help="--sweep channel counts (Cin = Cout)")
    ap.add_argument("--types", default=None,
                    help="--sweep conv types, comma list of: " + ", ".join(t[0] for t in CONV_TYPES))
    ap.add_argument("--out", default=None, help="output prefix: writes <out>_e<engines>.json and .png "
                    "(default scratchpad/conv_paths_sweep with --sweep, scratchpad/conv_paths_cases otherwise)")
    ap.add_argument("--plot-only", default=None, help="redraw the graph from a saved results JSON (no board)")
    ap.add_argument("--auto", action="store_true",
                    help="verify UnifiedEngine.conv2d_auto: routing table on the host, then each AUTO_CASES conv "
                         "on the board (1 engine) -- the wrapper's route and its result")
    ap.add_argument("--routes-only", action="store_true", help="with --auto: host routing table only, no board")
    ap.add_argument("--wino-budget-mb", type=int, default=None,
                    help="cap on Winograd D+M planes per engine (default 128); small values force tile-row groups")
    args = ap.parse_args()
    if args.wino_budget_mb:
        global WINO_BUDGET
        WINO_BUDGET = args.wino_budget_mb << 20

    if args.auto:
        t0 = time.perf_counter()
        routes_ok = check_routes()
        if args.routes_only:
            print("CONV_AUTO_ROUTES:", "PASS" if routes_ok else "FAIL")
            sys.exit(0 if routes_ok else 1)
        args.engines = 1
        probe = Probe(args)
        print("conv2d_auto on the board (1 engine, 12x20 maps):")
        results = []
        for sp in build_specs(args):
            results.extend(probe.run_case(sp, ["winograd", "direct"], auto=sp.auto))
        board_ok = all(r["status"] == "ok" for r in results)
        print(f"CONV_AUTO: {'PASS' if routes_ok and board_ok else 'FAIL'}  (routes {'PASS' if routes_ok else 'FAIL'}, "
              f"board {sum(r['status'] == 'ok' for r in results)}/{len(results)})  {time.perf_counter() - t0:.1f}s")
        sys.exit(0 if routes_ok and board_ok else 1)

    if args.plot_only:
        with open(args.plot_only) as f:
            saved = json.load(f)
        print_summary(saved["results"], saved.get("peak"))
        plot_all(saved["results"], os.path.splitext(args.plot_only)[0], saved["title"])
        return

    paths = (args.paths or ("winograd,native16,direct,native8" if args.sweep else "direct,winograd,native16,native8")).split(",")
    specs = build_specs(args)
    probe = Probe(args)
    out = args.out or os.path.join(ROOT, "scratchpad", "conv_paths_sweep" if args.sweep else "conv_paths_cases")
    prefix = f"{out}_e{args.engines}"
    title = (f"Conv lowering paths on {args.engines} engine{'s' if args.engines > 1 else ''} "
             f"({probe.peak:.0f} GFLOPS matmul peak), bf16 activations")
    results = []
    try:
        for sp in specs:
            results.extend(probe.run_case(sp, paths))
    finally:                                                 # keep whatever ran, even after a board error
        if results:
            os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
            with open(prefix + ".json", "w") as f:
                json.dump({"title": title, "peak": probe.peak, "args": vars(args), "results": results}, f, indent=1)
            print(f"results: {prefix}.json")
    print_summary(results, probe.peak)
    plot_all(results, prefix, title)


if __name__ == "__main__":
    main()
