"""YOLO Winograd F(2x2,3x3), adapted from winograd branch 7c7ff3289d9b.

The branch's SRAM transforms and sixteen BF16 matmuls are retained. Quantized
YOLO weights are dequantized offline before transformation; the CPU reference
continues to use the original quantized weights. This changes rounding, not
model parameters. Every operation stays in the resident FPGA program.
"""
from types import SimpleNamespace

import torch

import user_dma_core as core
from yolov5_common import QuantizedConv, dequantize_conv

SCRATCH_BYTES = 32 << 20
_CONV_SRAM_BANK = 0x80000
_CONV_STAGE = 4000 * 128


def _up(n, m):
    return (n + m - 1) // m * m


def eligible(operation, weight, source, destination, min_channels=128):
    return (operation["op"] == "conv"
            and tuple(weight["codes_shape"][-2:]) == (3, 3)
            and operation["stride"] == operation["dilation"] == 1
            and operation["pad"] == 1
            and min(source.channels, destination.channels) >= min_channels
            # At 128 channels an 8x8 map, or a small map with partial edge
            # tiles (13x13), loses to the native IF4 walker on the board.
            and (min(source.channels, destination.channels) >= 256
                 or (source.height * source.width >= 100
                     and (source.height * source.width >= 256
                          or source.height % 2 == source.width % 2 == 0)))
            and destination.logical_shape[1:] == source.logical_shape[1:])


def padded_bytes(source):
    # Odd output edges still need a complete 4x4 input tile. The additional
    # bottom/right zeros prevent those gathers from reading beyond the halo.
    return (_up(source.height, 2) + 2) * (_up(source.width, 2) + 2) * source.physical_channels * 2


def prepare(operation, weight, source, destination, image, unpack_codes):
    prepared = QuantizedConv(
        codes=unpack_codes(weight["codes_packed"], weight["codes_shape"], weight["precision"]),
        block_scales=weight["block_scales"], bias=None,
        data_type=core.TYPE[weight["precision"].upper()], gather=weight["layout"] == "gather")
    logical = dequantize_conv(prepared).float()
    ci, co = source.physical_channels, destination.physical_channels
    kernel = torch.zeros(co, ci, 3, 3)
    kernel[:destination.channels, list(source.channel_map)] = logical
    G = torch.tensor([[1., 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])
    transformed = torch.einsum("ik,ockl,jl->ocij", G, kernel, G)
    planes = []
    for r in range(4):
        for c in range(4):
            sign = (-1 if r == 2 else 1) * (-1 if c == 2 else 1)
            planes.append(image.allocate(
                (sign * transformed[:, :, r, c]).to(torch.bfloat16).contiguous(), alignment=128))
    bias = torch.zeros(co, dtype=torch.bfloat16)
    if weight["bias"].numel():
        bias[:destination.channels] = weight["bias"]
    return dict(us=[planes], b=image.allocate(bias, alignment=128),
                i64=image.allocate(torch.eye(64, dtype=torch.bfloat16), alignment=128),
                operation=operation, source=source, destination=destination)


class _ConvMap:
    def __init__(self, H, W, C, hh, hw, addr):
        self.C, self.hh, self.hw, self.addr, self.pb, self.Wp = C, hh, hw, addr, C * 2, W + 2 * hw
        self.pitch = self.Wp * self.pb

    def pix(self, y, x):
        return self.addr + ((y + self.hh) * self.Wp + x + self.hw) * self.pb


def _conv_wino_chunk(tw, ci, co):
    return next((t for t in range(tw, 0, -1) if 18 * _up(t * co * 2, 128) <= _CONV_SRAM_BANK
                 and 12 * _up((t + 1) * ci * 2, 128) <= _CONV_SRAM_BANK), None)


def emit(engine, plan, scratch, zero_address, zero_bytes):
    source, dest = plan["source"], plan["destination"]
    ci, co = source.physical_channels, dest.physical_channels
    size = padded_bytes(source)
    # Each invocation initializes its halo: scratch is shared with other ops.
    for offset in range(0, size, zero_bytes):
        take = min(zero_bytes, size - offset)
        engine.accelerator_memory_to_sram(zero_address, 0, take // 2)
        engine.sram_to_accelerator_memory(0, scratch + offset, take // 2)
    X = _ConvMap(_up(source.height, 2), _up(source.width, 2), ci, 1, 1, scratch)
    Y = _ConvMap(dest.height, dest.width, co, 0, 0, dest.address)
    for row in range(source.height):
        _conv_scopy(engine, source.address + row * source.width * ci * 2,
                    X.pix(row, 0), source.width, ci * 2)
    geometry = SimpleNamespace(Ci=ci, Co=co, ph=1, pw=1,
                               Hout=dest.height, Wout=dest.width,
                               wino=dict(gs=1, dim=2, nph=1, npw=1))
    gpr = engine.alloc_isa_reg()
    try:
        _conv_emit_wino(engine, plan, geometry, X, Y, (0, dest.height, 0, dest.width),
                        "silu" if plan["operation"]["activate"] else None,
                        scratch + size, SCRATCH_BYTES, gpr)
    finally:
        engine.release_isa_reg()


def _conv_scopy(e, src, dst, count, chunk, src_jump=0, dst_jump=0):
    src_jump, dst_jump = (0 if src_jump == chunk else src_jump), (0 if dst_jump == chunk else dst_jump)
    for i0 in range(0, count, max(1, _CONV_STAGE // chunk)):
        n = min(max(1, _CONV_STAGE // chunk), count - i0)
        e.accelerator_memory_to_sram(src + i0 * (src_jump or chunk), 0, n * chunk // 2,
                                     stride_bytes_per_chunk=chunk if src_jump else 0, stride_jump_bytes=src_jump)
        e.sram_to_accelerator_memory(0, dst + i0 * (dst_jump or chunk), n * chunk // 2,
                                     stride_bytes_per_chunk=chunk if dst_jump else 0, stride_jump_bytes=dst_jump)


def _conv_emit_wino(e, P, g, X, Y, band, act, scratch, scratch_bytes, gM):
    """F(2x2,3x3) / F(2,3): transforms as SRAM add/subs (A-bank op B-bank), matmuls with K = Cin."""
    gs, dim, nph, npw = (g.wino[k] for k in ("gs", "dim", "nph", "npw"))
    Ci, Co, step = g.Ci, g.Co, 2 * g.wino["gs"]
    ra, rb, ca, cb = band
    assert ca % step == 0 and (dim == 1 or ra % step == 0), f"Winograd band must start on a multiple of {step}"
    ta, tb = (ra // step, -(-rb // step)) if dim == 2 else (ra, rb)
    xa, xb = ca // step, -(-cb // step)
    twe, npl, nout = xb - xa, (16 if dim == 2 else 4), (4 if dim == 2 else 2)
    per_tile = npl * (Ci + Co) * 2
    if twe * per_tile > scratch_bytes:                                # D + M planes too wide: column groups
        tw_g = max(1, scratch_bytes // per_tile)
        for x in range(xa, xb, tw_g):
            _conv_emit_wino(e, P, g, X, Y, (ra, rb, x * step, min((x + tw_g) * step, cb)), act, scratch,
                            scratch_bytes, gM)
        return
    twc = _conv_wino_chunk(twe, Ci, Co)
    tg = max(1, min(tb - ta, scratch_bytes // (npl * twe * (Ci + Co) * 2)))
    D = [scratch + p * tg * twe * Ci * 2 for p in range(npl)]
    Mx = [scratch + npl * tg * twe * Ci * 2 + p * tg * twe * Co * 2 for p in range(npl)]
    A0, B0 = 0, _CONV_SRAM_BANK
    add, sub = e.eltwise_add_core, e.eltwise_sub_core
    sz, Po = _up((twc + 1) * X.pb, 128), _up(twc * Co * 2, 128)
    chunks = [(c0, min(twc, xb - c0)) for c0 in range(xa, xb, twc)]
    col0 = lambda c0, par, b, gb: gs * (2 * c0 + par + 3 * b) + gb - g.pw
    load = lambda iy, c0, par, b, gb, dst, ne: e.accelerator_memory_to_sram(
        X.pix(iy, col0(c0, par, b, gb)), dst, ne, stride_bytes_per_chunk=X.pb, stride_jump_bytes=2 * gs * X.pb)
    for ga in (range(gs) if dim == 2 else (0,)):
        for gb in range(gs):
            for g0 in range(ta, tb, tg):
                g1 = min(g0 + tg, tb)
                nt = (g1 - g0) * twe
                for pi, (a, b) in enumerate((a, b) for a in range(nph) for b in range(npw)):
                    for ty in range(g0, g1):
                        for c0, n in chunks:
                            off, ne, nel = (ty - g0) * twe + c0 - xa, (n + 1) * Ci, n * Ci
                            if dim == 2:                                # rows: D0-D2, D1+D2, D1-D2, D1-D3
                                DS = lambda r, p: (A0 if r < 2 else B0) + ((r % 2) * 2 + p) * sz
                                TA = lambda k, p: A0 + (4 + k * 2 + p) * sz
                                TB = lambda k, p: B0 + (4 + k * 2 + p) * sz
                                for r in range(4):
                                    for p in (0, 1):
                                        load(gs * (2 * ty + r + 3 * a) + ga - g.ph, c0, p, b, gb, DS(r, p), ne)
                                for T in (TA, TB):
                                    for p in (0, 1):
                                        sub(DS(0, p), DS(2, p), T(0, p), ne)
                                        add(DS(1, p), DS(2, p), T(1, p), ne)
                                        sub(DS(1, p), DS(2, p), T(2, p), ne)
                                        sub(DS(1, p), DS(3, p), T(3, p), ne)
                                rows, Vs = [(TA(k, 0), TA(k, 1), TB(k, 0), TB(k, 1), 4 * k) for k in range(4)], \
                                    [A0 + q * sz for q in range(4)]
                            else:
                                for dst, p in ((A0, 0), (A0 + sz, 1), (B0, 0), (B0 + sz, 1)):
                                    load(ty - g.ph, c0, p, b, gb, dst, ne)
                                rows, Vs = [(A0, A0 + sz, B0, B0 + sz, 0)], [A0 + (2 + q) * sz for q in range(4)]
                            for ae, ao, be, bo, p0 in rows:             # cols: d0-d2, d1+d2, d1-d2, d1-d3
                                sub(ae, be + X.pb, Vs[0], nel)
                                add(ao, be + X.pb, Vs[1], nel)
                                sub(ao, be + X.pb, Vs[2], nel)
                                sub(ao, bo + X.pb, Vs[3], nel)
                                for q in range(4):
                                    e.sram_to_accelerator_memory(Vs[q], D[p0 + q] + off * Ci * 2, nel)
                    for xi in range(npl):                               # pieces after the first accumulate
                        e.generate_instruction_add_set(gM, nt)
                        kw_ = (dict(C_DRAM_ADDR=Mx[xi], bias_mode="full_matrix") if pi else
                               dict(C_DRAM_ADDR=P["b"], bias_mode="broadcast_N") if xi == (5 if dim == 2 else 1)
                               else {})
                        e.matmat_mul_core(M=nt, K=Ci, N=Co, A_DRAM_ADDR=D[xi], B_DRAM_ADDR=P["us"][pi][xi],
                                          OUTPUT_DRAM_ADDR=Mx[xi], gpr_M_reg=gM, **kw_)
                for ty in range(g0, g1):                                # outputs: y0 = M0+M1+M2, y1 = M1-M2-M3
                    for c0, n in chunks:
                        off, nel = (ty - g0) * twe + c0 - xa, n * Co
                        if dim == 2:
                            idx = {0: 0, 2: 1, 3: 2}
                            Ms = lambda r, c: A0 + c * Po if r == 1 else B0 + (idx[r] * 4 + c) * Po
                            Ss = lambda i, c: A0 + (4 + i) * Po if c == 1 else B0 + (12 + i * 3 + idx[c]) * Po
                            T0, T1, Ys = A0 + 6 * Po, A0 + 7 * Po, (lambda k: A0 + (8 + k) * Po)
                            for r in range(4):
                                for c in range(4):
                                    e.accelerator_memory_to_sram(Mx[r * 4 + c] + off * Co * 2, Ms(r, c), nel)
                            for c in range(4):
                                add(Ms(1, c), Ms(2, c), T0, nel)
                                add(T0, Ms(0, c), Ss(0, c), nel)
                                sub(Ms(1, c), Ms(2, c), T1, nel)
                                sub(T1, Ms(3, c), Ss(1, c), nel)
                            for i in range(2):
                                add(Ss(i, 1), Ss(i, 2), T0, nel)
                                add(T0, Ss(i, 0), Ys(2 * i), nel)
                                sub(Ss(i, 1), Ss(i, 2), T1, nel)
                                sub(T1, Ss(i, 3), Ys(2 * i + 1), nel)
                        else:
                            M1, M0, M2, M3, T0, T1 = A0, B0, B0 + Po, B0 + 2 * Po, A0 + Po, A0 + 2 * Po
                            Ys = lambda k: A0 + (3 + k) * Po
                            for p, dst in ((1, M1), (0, M0), (2, M2), (3, M3)):
                                e.accelerator_memory_to_sram(Mx[p] + off * Co * 2, dst, nel)
                            add(M1, M2, T0, nel)
                            add(T0, M0, Ys(0), nel)
                            sub(M1, M2, T1, nel)
                            sub(T1, M3, Ys(1), nel)
                        for k in range(nout):
                            e.sram_to_accelerator_memory(Ys(k), Mx[k] + off * Co * 2, nel)
                if act:
                    for yp in Mx[:nout]:
                        e.generate_instruction_add_set(gM, nt * Co // 64)
                        e.activation_core(M=nt * Co // 64, N=64, A_DRAM_ADDR=yp, OUTPUT_DRAM_ADDR=yp,
                                          IDENTITY_DRAM_ADDR=P["i64"], activation="clamp" if act == "relu" else act,
                                          gpr_M_reg=gM)
                for k in range(nout):                                   # scatter valid outputs only
                    i, j = (k // 2, k % 2) if dim == 2 else (0, k)
                    x0 = gs * (2 * xa + j) + gb
                    cnt = min(twe, max(0, -(-(g.Wout - x0) // step)))
                    for ty in range(g0, g1):
                        oy = gs * (2 * ty + i) + ga if dim == 2 else ty
                        if cnt and oy < g.Hout:
                            _conv_scopy(e, Mx[k] + (ty - g0) * twe * Co * 2, Y.pix(oy, x0), cnt, Co * 2,
                                        dst_jump=step * Y.pb)
