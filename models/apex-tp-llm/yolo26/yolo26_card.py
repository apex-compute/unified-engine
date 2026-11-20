"""YOLO26-OBB on one Apex engine: graph lowering, quantization, memory plan and a sliced program.

The graph is lowered to the vendor's convolution / max-pool / upsample / add / concat emitters
(models/yolov5s/yolov5_precompiled.py) plus an attention block built from matmul + softmax.
What differs from the vendor's YOLOv5 artifact:
  * relocatable: the parameter + program image and the tensors go to DRAM ranges the caller picks
    (the free regions next to an LLM on the same engine)
  * a tensor buffer is reused once its last reader has run
  * the program is cut into short units that each end in HALT, so it can run in slices between
    another program's segments on the same engine: no unit uses a GPR, and every unit reloads the
    SRAM state it needs (scales, bias)
  * depthwise 3x3 convolutions run as dense convolutions with diagonal weights (exact, no new primitive)
  * the three final 1x1 convolutions of each head level are one convolution: 4 box + 15 class + 1 angle
"""

import math
import os
import sys

import torch
import torch.nn as nn

UE_ROOT = os.path.expanduser(os.environ.get("UE_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
for p in (os.path.join(UE_ROOT, "models", "yolov5s"), UE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import user_dma_core as U          # noqa: E402
import yolov5_common as VC         # noqa: E402
import yolov5_precompiled as VP    # noqa: E402
import yolov5_artifact as VA       # noqa: E402

LINE = VP._LINE_BYTES              # one 64-lane bf16 line
VEC = U.UE_VECTOR_SIZE
UNIT_S = float(os.environ.get("YOLO_UNIT_MS", "4")) / 1e3   # target card time of one program unit
MACS_PER_S = float(os.environ.get("YOLO_MACS_PER_S", "10e9"))  # for the unit estimate only
MASK_NEG = -30000.0
# the CONV X stream repeats a layer's weights once per pixel of its tile, so this bounds both the
# image size (sum over layers) and the tile size (pixels per launch)
STREAM_BUDGET = int(float(os.environ.get("YOLO_STREAM_KB", "1024")) * 1024)


def align(v, a):
    return (int(v) + a - 1) // a * a


# ----------------------------------------------------------------------------------------------
# graph lowering: ultralytics modules -> a flat list of primitive operations

class Graph:
    def __init__(self, H, W):
        self.ops = []
        self.shape = {"input": (3, H, W)}
        self.n = 0

    def _name(self, base):
        self.n += 1
        return "%s#%d" % (base, self.n)

    def conv(self, base, x, w, b, stride=1, pad=0, act=False):
        oc, ic, kh, kw = w.shape
        c, h, wd = self.shape[x]
        assert ic == c, (base, ic, c)
        name = self._name(base)
        oh, ow = (h + 2 * pad - kh) // stride + 1, (wd + 2 * pad - kw) // stride + 1
        self.ops.append({"op": "conv", "name": name, "inputs": [x], "output": name, "w": w.float(),
                         "b": b.float(), "stride": stride, "pad": pad, "dilation": 1, "activate": bool(act)})
        self.shape[name] = (oc, oh, ow)
        return name

    def maxpool(self, base, x, k, s, p):
        c, h, w = self.shape[x]
        name = self._name(base)
        self.ops.append({"op": "maxpool", "name": name, "inputs": [x], "output": name,
                         "kernel": k, "stride": s, "pad": p})
        self.shape[name] = (c, (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1)
        return name

    def upsample(self, base, x):
        c, h, w = self.shape[x]
        name = self._name(base)
        self.ops.append({"op": "upsample2x", "name": name, "inputs": [x], "output": name})
        self.shape[name] = (c, 2 * h, 2 * w)
        return name

    def add(self, base, a, b):
        assert self.shape[a] == self.shape[b], (base, self.shape[a], self.shape[b])
        name = self._name(base)
        self.ops.append({"op": "add", "name": name, "inputs": [a, b], "output": name})
        self.shape[name] = self.shape[a]
        return name

    def concat(self, base, xs):
        name = self._name(base)
        hw = self.shape[xs[0]][1:]
        assert all(self.shape[x][1:] == hw for x in xs)
        self.ops.append({"op": "concat", "name": name, "inputs": list(xs), "output": name})
        self.shape[name] = (sum(self.shape[x][0] for x in xs), *hw)
        return name

    def attention(self, base, q, k, v):
        c, h, w = self.shape[v]
        name = self._name(base)
        self.ops.append({"op": "attention", "name": name, "inputs": [q, k, v], "output": name})
        self.shape[name] = (c, h, w)
        return name


def fold(m, rows=None, scale=1.0):
    """Conv(+BN) of an ultralytics Conv / nn.Conv2d -> dense float weight, bias (depthwise made dense)."""
    conv = m.conv if hasattr(m, "conv") else m
    bn = getattr(m, "bn", None)
    w, b = VC.fold_conv_bn(conv, bn)
    if b is None:
        b = torch.zeros(conv.out_channels)
    if conv.groups != 1:
        assert conv.groups == conv.in_channels == conv.out_channels, "only depthwise grouping"
        d = torch.zeros(conv.out_channels, conv.in_channels, *w.shape[2:])
        idx = torch.arange(conv.out_channels)
        d[idx, idx] = w[:, 0]
        w = d
    if rows is not None:
        w, b = w[rows], b[rows]
    return w * scale, b * scale


def act_of(m):
    a = getattr(m, "act", None)
    if a is None or type(a).__name__ == "Identity":
        return False
    assert type(a).__name__ == "SiLU", type(a).__name__
    return True


def kpad(m):
    conv = m.conv if hasattr(m, "conv") else m
    return int(conv.stride[0]), int(conv.padding[0])


def L_conv(g, m, x, base, rows=None, scale=1.0, act=None):
    w, b = fold(m, rows, scale)
    s, p = kpad(m)
    return g.conv(base, x, w, b, s, p, act_of(m) if act is None else act)


def L_bottleneck(g, m, x, base):
    t = L_conv(g, m.cv1, x, base + ".cv1")
    t = L_conv(g, m.cv2, t, base + ".cv2")
    return g.add(base + ".add", x, t) if m.add else t


def L_c3(g, m, x, base):
    a = L_conv(g, m.cv1, x, base + ".cv1")
    for i, bm in enumerate(m.m):
        a = L_bottleneck(g, bm, a, "%s.m%d" % (base, i))
    bb = L_conv(g, m.cv2, x, base + ".cv2")
    return L_conv(g, m.cv3, g.concat(base + ".cat", [a, bb]), base + ".cv3")


def L_attention(g, m, x, base):
    nh, kd, hd = m.num_heads, m.key_dim, m.head_dim
    q, k, v = [], [], []
    for h in range(nh):
        o = h * (2 * kd + hd)
        q.append(L_conv(g, m.qkv, x, "%s.q%d" % (base, h), rows=slice(o, o + kd), scale=m.scale))
        k.append(L_conv(g, m.qkv, x, "%s.k%d" % (base, h), rows=slice(o + kd, o + 2 * kd)))
        v.append(L_conv(g, m.qkv, x, "%s.v%d" % (base, h), rows=slice(o + 2 * kd, o + 2 * kd + hd)))
    outs = [g.attention("%s.sdpa%d" % (base, h), q[h], k[h], v[h]) for h in range(nh)]
    o = g.concat(base + ".ocat", outs) if nh > 1 else outs[0]
    vv = g.concat(base + ".vcat", v) if nh > 1 else v[0]
    pe = L_conv(g, m.pe, vv, base + ".pe")
    return L_conv(g, m.proj, g.add(base + ".ope", o, pe), base + ".proj")


def L_psa(g, m, x, base):
    a = L_attention(g, m.attn, x, base + ".attn")
    x = g.add(base + ".add1", x, a) if m.add else a
    f = L_conv(g, m.ffn[0], x, base + ".ffn0")
    f = L_conv(g, m.ffn[1], f, base + ".ffn1")
    return g.add(base + ".add2", x, f) if m.add else f


def L_inner(g, m, x, base):
    kind = type(m).__name__
    if kind == "Bottleneck":
        return L_bottleneck(g, m, x, base)
    if kind == "C3k":
        return L_c3(g, m, x, base)
    if kind == "PSABlock":
        return L_psa(g, m, x, base)
    if kind == "Sequential":
        for i, s in enumerate(m):
            x = L_inner(g, s, x, "%s.%d" % (base, i))
        return x
    raise ValueError("unsupported block %s" % kind)


def L_c2f(g, m, x, base):
    c = m.c
    ys = [L_conv(g, m.cv1, x, base + ".cv1a", rows=slice(0, c)),
          L_conv(g, m.cv1, x, base + ".cv1b", rows=slice(c, 2 * c))]
    for i, bm in enumerate(m.m):
        ys.append(L_inner(g, bm, ys[-1], "%s.m%d" % (base, i)))
    return L_conv(g, m.cv2, g.concat(base + ".cat", ys), base + ".cv2")


def L_sppf(g, m, x, base):
    y = [L_conv(g, m.cv1, x, base + ".cv1")]
    k, s, p = (int(v if not isinstance(v, (tuple, list)) else v[0]) for v in (m.m.kernel_size, m.m.stride, m.m.padding))
    for i in range(getattr(m, "n", 3)):
        y.append(g.maxpool("%s.mp%d" % (base, i), y[-1], k, s, p))
    out = L_conv(g, m.cv2, g.concat(base + ".cat", y), base + ".cv2")
    return g.add(base + ".add", out, x) if getattr(m, "add", False) else out


def L_c2psa(g, m, x, base):
    c = m.c
    a = L_conv(g, m.cv1, x, base + ".cv1a", rows=slice(0, c))
    b = L_conv(g, m.cv1, x, base + ".cv1b", rows=slice(c, 2 * c))
    for i, pm in enumerate(m.m):
        b = L_psa(g, pm, b, "%s.m%d" % (base, i))
    return L_conv(g, m.cv2, g.concat(base + ".cat", [a, b]), base + ".cv2")


def L_head(g, det, feats):
    """one2one branches of OBB26; per level one output map: 4 box distances, nc class logits, 1 angle."""
    outs = []
    for i, x in enumerate(feats):
        base = "head%d" % i
        bx = det.one2one_cv2[i]
        cl = det.one2one_cv3[i]
        an = det.one2one_cv4[i]
        t = L_conv(g, bx[0], x, base + ".box0")
        t = L_conv(g, bx[1], t, base + ".box1")
        c = L_conv(g, cl[0][0], x, base + ".cls0dw")
        c = L_conv(g, cl[0][1], c, base + ".cls0")
        c = L_conv(g, cl[1][0], c, base + ".cls1dw")
        c = L_conv(g, cl[1][1], c, base + ".cls1")
        a = L_conv(g, an[0], x, base + ".ang0")
        a = L_conv(g, an[1], a, base + ".ang1")
        cat = g.concat(base + ".cat", [t, c, a])
        wb, bb = fold(bx[2])
        wc, bc = fold(cl[2])
        wa, ba = fold(an[2])
        ct, cc, ca = g.shape[t][0], g.shape[c][0], g.shape[a][0]
        nb, nc, na = wb.shape[0], wc.shape[0], wa.shape[0]
        w = torch.zeros(nb + nc + na, ct + cc + ca, 1, 1)
        w[:nb, :ct] = wb
        w[nb:nb + nc, ct:ct + cc] = wc
        w[nb + nc:, ct + cc:] = wa
        outs.append(g.conv(base + ".out", cat, w, torch.cat([bb, bc, ba])))
    return outs


def stem_s2d(m):
    """The 3x3 stride-2 pad-1 stem as a 3x3 stride-1 pad-1 conv on the 2x2 space-to-depth input.
    s2d channel (a, b, c) = a*6 + b*3 + c holds pixel (2I + a, 2J + b) of colour c. A kernel row
    dy = -1 / 0 / +1 reads s2d row I-1 (a=1) / I (a=0) / I (a=1); the I+1 row stays zero."""
    w, b = fold(m)
    oc = w.shape[0]
    w2 = torch.zeros(oc, 12, 3, 3)
    pick = {0: (0, 1), 1: (1, 0), 2: (1, 1)}       # kernel index -> (s2d offset index, phase)
    for ky in range(3):
        oy, a = pick[ky]
        for kx in range(3):
            ox, bb = pick[kx]
            for c in range(3):
                w2[:, a * 6 + bb * 3 + c, oy, ox] = w[:, c, ky, kx]
    return w2, b


def lower(model, H, W):
    g = Graph(H, W)
    g.shape["input"] = (12, H // 2, W // 2)       # the host uploads the 2x2 space-to-depth image
    vals = []
    cur = "input"
    for i, m in enumerate(model.model):
        kind = type(m).__name__
        src = [cur if f == -1 else vals[f] for f in (m.f if isinstance(m.f, list) else [m.f])]
        base = "m%d" % i
        if i == 0:
            conv = m.conv
            assert kind == "Conv" and conv.in_channels == 3 and conv.kernel_size == (3, 3) \
                and conv.stride == (2, 2) and conv.padding == (1, 1)
            w2, b2 = stem_s2d(m)
            out = g.conv(base, "input", w2, b2, 1, 1, act_of(m))
        elif kind == "Conv":
            out = L_conv(g, m, src[0], base)
        elif kind == "C3k2":
            out = L_c2f(g, m, src[0], base)
        elif kind == "SPPF":
            out = L_sppf(g, m, src[0], base)
        elif kind == "C2PSA":
            out = L_c2psa(g, m, src[0], base)
        elif kind == "Upsample":
            assert m.mode == "nearest" and float(m.scale_factor) == 2.0
            out = g.upsample(base, src[0])
        elif kind == "Concat":
            out = g.concat(base, src)
        elif kind in ("OBB26", "OBB"):
            heads = L_head(g, m, src)
            return g, heads, [float(s) for s in m.stride], int(m.nc)
        else:
            raise ValueError("unsupported module %s" % kind)
        vals.append(out)
        cur = out
    raise RuntimeError("no head")


# ----------------------------------------------------------------------------------------------
# memory plan

class Arena:
    """First-fit allocator over [base, limit)."""

    def __init__(self, base, limit):
        self.base, self.limit = int(base), int(limit)
        self.free = [(self.base, self.limit)]
        self.peak = self.base

    def alloc(self, size, a=LINE):
        size = align(size, a)
        for i, (s, e) in enumerate(self.free):
            s2 = align(s, a)
            if s2 + size <= e:
                rest = []
                if s2 > s:
                    rest.append((s, s2))
                if s2 + size < e:
                    rest.append((s2 + size, e))
                self.free[i:i + 1] = rest
                self.peak = max(self.peak, s2 + size)
                return s2
        raise RuntimeError("tensor arena full: need %d bytes, arena %d MB, peak %.1f MB"
                           % (size, (self.limit - self.base) >> 20, (self.peak - self.base) / 1e6))

    def release(self, addr, size, a=LINE):
        size = align(size, a)
        self.free.append((addr, addr + size))
        self.free.sort()
        merged = []
        for s, e in self.free:
            if merged and merged[-1][1] >= s:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        self.free = merged


def plan_layouts(g, heads):
    lay = {"input": VP._PackedMap(g.shape["input"], VEC, tuple(range(g.shape["input"][0])))}
    rows = {}
    for op in g.ops:
        shape = g.shape[op["output"]]
        src = [lay[n] for n in op["inputs"]]
        if op["op"] == "conv":
            lay[op["output"]] = VP._PackedMap(shape, align(shape[0], VEC), tuple(range(shape[0])))
        elif op["op"] in ("maxpool", "upsample2x"):
            lay[op["output"]] = VP._PackedMap(shape, src[0].physical_channels, tuple(src[0].channel_map))
        elif op["op"] == "add":
            a, b = src
            assert a.physical_channels == b.physical_channels and a.channel_map == b.channel_map, op["name"]
            lay[op["output"]] = VP._PackedMap(shape, a.physical_channels, tuple(a.channel_map))
        elif op["op"] == "concat":
            cmap, off = [], 0
            for s in src:
                cmap.extend(off + v for v in s.channel_map)
                off += s.physical_channels
            lay[op["output"]] = VP._PackedMap(shape, off, tuple(cmap))
        elif op["op"] == "attention":
            q, k, v = src
            assert q.physical_channels == k.physical_channels == VEC and v.physical_channels == VEC, op["name"]
            n = shape[1] * shape[2]
            npad = align(n, VEC)
            rows[op["inputs"][1]] = npad          # keys and values get zeroed tail rows up to npad
            rows[op["inputs"][2]] = npad
            lay[op["output"]] = VP._PackedMap(shape, VEC, tuple(range(shape[0])))
    return lay, rows


def tensor_bytes(m, rows=None):
    return (rows or m.height * m.width) * m.physical_channels * 2


# ----------------------------------------------------------------------------------------------
# quantization

PRECISION = os.environ.get("YOLO_PREC", "if8")      # channel-layout convolutions: if8 or if4


def quantize_channels_if8(conv):
    """quantize_conv_if4's [oc, ky, kx, ct] 64-value blocks with IF8 codes (one byte per weight)."""
    weight = conv.weight.detach().float()
    oc, channels, kh, kw = weight.shape
    ct = (channels + VEC - 1) // VEC
    padded = torch.zeros(oc, ct * VEC, kh, kw)
    padded[:, :channels] = weight
    blocks = padded.view(oc, ct, VEC, kh, kw).permute(0, 3, 4, 1, 2).contiguous().view(-1, VEC)
    data, scale = VC.quant_lib.quantize_if8(blocks, block_size=VEC, int_variant=None)
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(-1, VEC)
    codes = (raw.view(oc, kh, kw, ct, VEC).permute(0, 3, 4, 1, 2).contiguous()
             .view(oc, ct * VEC, kh, kw)[:, :channels].contiguous())
    scales = VC._bf16_from_bytes(scale).view(oc, kh * kw * ct).contiguous()
    return VC.QuantizedConv(codes=codes, block_scales=scales, bias=conv.bias.detach().to(torch.bfloat16).contiguous(),
                            data_type=U.TYPE.IF8, gather=False)


def conv_weight_entry(op, src_map, input_is_image):
    """Quantize a conv for the vendor plan. Weights are first spread over the input's physical channels
    (so a concat of padded pieces never mixes two scale blocks), except for the image stem (gather path)."""
    w, b = op["w"], op["b"]
    oc, ic, kh, kw = w.shape
    if input_is_image:
        wp, cin = w, ic
    else:
        cin = src_map.physical_channels
        wp = torch.zeros(oc, cin, kh, kw)
        for i, p in enumerate(src_map.channel_map):
            wp[:, p] = w[:, i]
    conv = nn.Conv2d(cin, oc, kh, stride=op["stride"], padding=op["pad"], bias=True)
    with torch.no_grad():
        conv.weight.copy_(wp)
        conv.bias.copy_(b)
    if PRECISION == "if8" and not VC._gather_is_profitable(conv):
        prep = quantize_channels_if8(conv)
    else:
        prep = VC.quantize_conv_for_andromeda(conv, None)
    precision = prep.data_type.name.lower()
    packed, shape = VA._pack_codes(prep.codes, precision)
    return {"precision": precision, "layout": "gather" if prep.gather else "channels",
            "codes_packed": packed, "codes_shape": shape,
            "block_scales": prep.block_scales.to(torch.bfloat16).contiguous(),
            "bias": prep.bias.contiguous()}


# ----------------------------------------------------------------------------------------------
# emission

class Units:
    """Cuts the captured instruction stream into HALT-terminated units of about UNIT_S card time."""

    def __init__(self, eng):
        self.eng = eng
        self.est = 0.0
        self.ests = []
        self.names = []
        self.cur = set()

    def charge(self, seconds, name):
        self.est += seconds
        self.cur.add(name.split("#")[0])

    def full(self):
        return self.est >= UNIT_S

    def cut(self):
        if not self.est:
            return
        self.eng.generate_instruction_halt()
        self.ests.append(self.est)
        self.names.append(",".join(sorted(self.cur)))
        self.est = 0.0
        self.cur = set()


def emit_conv(eng, plan, zero, units):
    """vendor _emit_conv with unit cuts between tiles; each unit starts by (re)loading scales / bias."""
    op = plan["operation"]
    src, dst = plan["source"], plan["destination"]
    stride, pad, dil = int(op["stride"]), int(op["pad"]), int(op["dilation"])
    lalu_mode, lalu_a, lalu_b = U._conv_fused_lalu(False, bool(op["activate"]), False)
    ct = (plan["convolution_c"] + VEC - 1) // VEC
    taps = plan["kernel_h"] * plan["kernel_w"] * ct
    scale_count = plan["oc_chunk"] * (plan["gather_chunks"] if plan["use_gather"] else taps)
    for chunk in plan["chunks"]:
        for start, stop, th, tw, _wh, win_w in plan["groups"]:
            loaded = False
            for tile in plan["tiles"][start:stop]:
                if not loaded:
                    eng.accelerator_memory_to_scale_sram(chunk["scale_address"], scale_count)
                    if chunk["bias_address"] is not None:
                        eng.accelerator_memory_to_bias_sram(chunk["bias_address"], plan["oc_chunk"])
                    loaded = True
                VP._stage_conv_window(eng, src, tile, pad, zero)
                eng.start_queue_for_conv2d_operation(
                    act_sram_start_addr=0, output_sram_wb_addr=VP._WB_SRAM_ADDRESS,
                    weights_dram_addr=chunk["weight_address"],
                    kernel_w=plan["kernel_w"], kernel_h=plan["kernel_h"], ct=ct, oc_count=plan["oc_chunk"],
                    out_w=tw, out_h=th, w_pad=win_w, stride_s=stride, data_type=plan["data_type"],
                    bias_enable=plan["bias_enabled"], lalu_mode=lalu_mode, lalu_a=lalu_a, lalu_b=lalu_b,
                    dilation=dil, gather=plan["use_gather"], c_in=plan["convolution_c"])
                VP._scatter_conv_tile(eng, dst, tile, chunk["oc0"], plan["oc_chunk"])
                ty, tx = int(tile[2]), int(tile[3])
                units.charge(ty * tx * plan["oc_chunk"] * taps * VEC / MACS_PER_S + 2e-5, op["name"])
                if units.full():
                    units.cut()
                    loaded = False


def zero_rows(eng, addr, nbytes, zero):
    done = 0
    while done < nbytes:
        take = min(nbytes - done, VP._ACT_TEMPLATE_BYTES)
        eng.accelerator_memory_to_sram(zero, 0, 0, memcpy_length_bytes=take)
        eng.sram_to_accelerator_memory(0, addr + done, 0, memcpy_length_bytes=take)
        done += take


def emit_attention(eng, q, k, v, out, n, npad, s_addr, vt_addr, mask, ident, zero):
    if npad > n:
        zero_rows(eng, k.address + n * LINE, (npad - n) * LINE, zero)
        zero_rows(eng, v.address + n * LINE, (npad - n) * LINE, zero)
    eng.bf16_transpose_core(npad, VEC, v.address, vt_addr, ident)                 # V^T: [64, npad]
    eng.matmat_mul_core(n, VEC, npad, q.address, k.address, s_addr, softmax_enable=True,
                        C_DRAM_ADDR=mask, bias_mode="broadcast_N")                 # softmax(q k^T + mask)
    eng.matmat_mul_core(n, npad, VEC, s_addr, vt_addr, out.address)                # P V


def compile_engine(model, H, W, *, pbase, plimit, tbase, tlimit, log=print):
    """Everything for one engine: returns the deployment image (params + program) and the run manifest."""
    prev = U.UE_AXI_DATA_WIDTH_BITS
    U.UE_AXI_DATA_WIDTH_BITS = VP.PRECOMPILED_AXI_DATA_WIDTH_BITS     # the cards' AXI width (HW_INFO: 256)
    try:
        return _compile_engine(model, H, W, pbase=pbase, plimit=plimit, tbase=tbase, tlimit=tlimit, log=log)
    finally:
        U.UE_AXI_DATA_WIDTH_BITS = prev


def _compile_engine(model, H, W, *, pbase, plimit, tbase, tlimit, log=print):
    g, heads, strides, nc = lower(model, H, W)
    lay, rows = plan_layouts(g, heads)

    image = VP._ImageBuilder(pbase, plimit)
    zero = image.allocate(torch.zeros(VP._ACT_TEMPLATE_BYTES // 2, dtype=torch.bfloat16))
    neg_inf = image.allocate(torch.full((VP._ACT_TEMPLATE_BYTES // 2,), float("-inf"), dtype=torch.bfloat16))
    ident = image.allocate(torch.eye(VEC, dtype=torch.bfloat16))
    masks = {}

    # tensors: input, the head bundle and the upsample scratch stay put; the rest is reused
    ar = Arena(tbase, tlimit)
    lay["input"].address = ar.alloc(tensor_bytes(lay["input"]))
    hb = 0
    for h in heads:
        hb += align(tensor_bytes(lay[h]), LINE)
    head_base = ar.alloc(hb)
    off = head_base
    for h in heads:
        lay[h].address = off
        off += align(tensor_bytes(lay[h]), LINE)
    up_scratch = 0
    for op in g.ops:
        if op["op"] == "upsample2x":
            s = lay[op["inputs"][0]]
            up_scratch = max(up_scratch, 2 * s.height * s.width * s.channel_tiles * LINE)
    scratch = ar.alloc(up_scratch) if up_scratch else 0

    last = {}
    for i, op in enumerate(g.ops):
        for n in op["inputs"]:
            last[n] = i
    pinned = set(heads) | {"input"}

    # weights first (they are part of the image), then the program behind them
    plans = {}
    for op in g.ops:
        if op["op"] != "conv":
            continue
        src = lay[op["inputs"][0]]
        is_img = op["inputs"][0] == "input"
        entry = conv_weight_entry(op, src, is_img)
        vsrc = src if is_img else VP._PackedMap(
            (src.physical_channels, src.height, src.width), src.physical_channels, tuple(range(src.physical_channels)))
        dst = lay[op["output"]]
        plans[op["name"]] = (entry, vsrc, dst)
    for name, (entry, vsrc, dst) in plans.items():
        op = next(o for o in g.ops if o["name"] == name)
        vop = {"name": name, "stride": op["stride"], "pad": op["pad"], "dilation": 1, "activate": op["activate"]}
        plans[name] = VP._prepare_conv_plan(vop, entry, vsrc, dst, image,
                                            weight_stream_budget_bytes=STREAM_BUDGET)
        plans[name]["source"] = vsrc
    for op in g.ops:
        if op["op"] == "attention":
            c, h, w = g.shape[op["output"]]
            n = h * w
            npad = align(n, VEC)
            if npad not in masks:
                m = torch.zeros(npad, dtype=torch.bfloat16)
                m[n:] = MASK_NEG
                masks[npad] = image.allocate(m)

    program_address = image.align(64)
    eng = VP._WholeGraphEngine(program_address)
    eng._tensor_dram_base = eng._tensor_dram_addr = tlimit      # nothing may allocate behind our back
    eng._params_dram_base = eng._next_params_dram_addr = plimit
    units = Units(eng)
    cut_each = bool(int(os.environ.get("YOLO_CUT_EACH_OP", "0")))     # debug: every op ends a unit
    trace = []
    eng.start_capture()
    for i, op in enumerate(g.ops):
        out = lay[op["output"]]
        if op["output"] not in pinned:
            out.address = ar.alloc(tensor_bytes(out, rows.get(op["output"])))
        trace.append({"name": op["name"], "op": op["op"], "address": out.address, "shape": list(out.logical_shape),
                      "physical": out.physical_channels, "channel_map": list(out.channel_map)})
        src = [lay[n] for n in op["inputs"]]
        kind = op["op"]
        if kind == "conv":
            plans[op["name"]]["source"].address = src[0].address      # the view was made before allocation
            emit_conv(eng, plans[op["name"]], zero, units)
        elif kind == "maxpool":
            VP._emit_maxpool(eng, op, src[0], out, neg_inf)
            units.charge(tensor_bytes(src[0]) * 6 / 2e9, op["name"])
        elif kind == "upsample2x":
            VP._emit_upsample(eng, src[0], out, scratch)
            units.charge(tensor_bytes(out) * 3 / 2e9, op["name"])
        elif kind == "add":
            VP._emit_add(eng, src[0], src[1], out)
            units.charge(tensor_bytes(out) * 3 / 2e9, op["name"])
        elif kind == "concat":
            VP._emit_concat(eng, src, out)
            units.charge(tensor_bytes(out) * 2 / 2e9, op["name"])
        elif kind == "attention":
            c, h, w = g.shape[op["output"]]
            n = h * w
            npad = align(n, VEC)
            s_addr = ar.alloc(n * npad * 2)
            vt_addr = ar.alloc(VEC * npad * 2)
            emit_attention(eng, src[0], src[1], src[2], out, n, npad, s_addr, vt_addr, masks[npad], ident, zero)
            ar.release(s_addr, n * npad * 2)
            ar.release(vt_addr, VEC * npad * 2)
            units.charge((n * npad * VEC * 2) / 3e9 + 3e-4, op["name"])
        else:
            raise AssertionError(kind)
        if units.full() or cut_each:
            units.cut()
        trace[-1]["unit"] = len(units.ests) - 1 if cut_each else None
        for n in set(op["inputs"]):
            if last[n] == i and n not in pinned:
                lay[n].address and ar.release(lay[n].address, tensor_bytes(lay[n], rows.get(n)))
    units.cut()
    eng.stop_capture()
    if eng._tensor_dram_addr != tlimit or eng._next_params_dram_addr != plimit:
        raise RuntimeError("an emitter allocated card memory on its own")

    program = b"".join(ins.get_bytes() for ins in eng.capture_buffer)
    types = [(int.from_bytes(program[o:o + 4], "little") >> 8) & 0xF for o in range(0, len(program), 32)]
    halts = [i for i, t in enumerate(types) if t == U.INSTRUCTION_HALT]
    starts = [0] + [(i + 1) * 32 for i in halts[:-1]]
    if halts[-1] != len(types) - 1 and any(t != U.INSTRUCTION_NOP for t in types[halts[-1] + 1:]):
        raise RuntimeError("program does not end in HALT")
    # only datapath ops, CONFIG, HALT and NOP: no register ALU / PBI set (GPRs of the LLM program
    # sharing the engine), no jumps, flags or software interrupts
    bad = set(types) - {U.INSTRUCTION_UE_OP, U.INSTRUCTION_CONFIG, U.INSTRUCTION_HALT, U.INSTRUCTION_NOP}
    if bad:
        raise RuntimeError("program uses instruction types %s" % sorted(hex(t) for t in bad))
    image.write(program_address, program)
    hb_map = []
    for h in heads:
        m = lay[h]
        hb_map.append({"offset": m.address - head_base, "shape": list(m.logical_shape),
                       "physical": m.physical_channels, "bytes": tensor_bytes(m)})
    man = {
        "H": H, "W": W, "nc": nc, "strides": strides,
        "image_base": pbase, "image_bytes": len(image.data),
        "program_address": program_address, "program_bytes": len(program),
        "unit_offsets": starts, "unit_est_s": units.ests, "unit_names": units.names,
        "input_address": lay["input"].address, "input_bytes": tensor_bytes(lay["input"]),
        "head_address": head_base, "head_bytes": hb, "heads": hb_map,
        "tensor_base": tbase, "tensor_peak": ar.peak - tbase,
        "ops": len(g.ops), "convs": len(plans), "trace": trace,
    }
    compile_engine.last_graph = g
    log("YOLO26 %dx%d: %d ops, %d convs, image %.1f MB (program %.0f KB, %d units, est %.0f ms), tensors %.1f MB"
        % (W, H, man["ops"], man["convs"], man["image_bytes"] / 1e6, man["program_bytes"] / 1e3,
           len(starts), sum(units.ests) * 1e3, man["tensor_peak"] / 1e6))
    return bytes(image.data), man


# ----------------------------------------------------------------------------------------------
# host side: input packing, output decode

def pack_input(chw):
    """[3, H, W] float in 0..1 -> 2x2 space-to-depth [12, H/2, W/2] -> the card's packed input map
    (one 64-lane bf16 line per pixel; see stem_s2d for the channel order)."""
    c, h, w = chw.shape
    s2d = chw.reshape(c, h // 2, 2, w // 2, 2).permute(2, 4, 0, 1, 3).reshape(4 * c, h // 2, w // 2)
    return U.conv2d_pack_activation_map(s2d.to(torch.bfloat16), 0)


def decode(man, bundle, conf=0.25, max_det=300):
    """bundle: bf16 head maps as read back -> list of (cx, cy, w, h, angle, score, class) in input pixels."""
    nc = man["nc"]
    rows, anchors, strides = [], [], []
    for h, s in zip(man["heads"], man["strides"]):
        c, hh, ww = h["shape"]
        a = bundle[h["offset"] // 2:(h["offset"] + h["bytes"]) // 2].float().view(hh * ww, h["physical"])[:, :c]
        rows.append(a)
        ys, xs = torch.meshgrid(torch.arange(hh) + 0.5, torch.arange(ww) + 0.5, indexing="ij")
        anchors.append(torch.stack((xs.reshape(-1), ys.reshape(-1)), 1))
        strides.append(torch.full((hh * ww, 1), s))
    p = torch.cat(rows)
    anc = torch.cat(anchors)
    st = torch.cat(strides)
    dist, logits, ang = p[:, :4], p[:, 4:4 + nc], p[:, 4 + nc:5 + nc]
    lt, rb = dist[:, :2], dist[:, 2:4]
    cos, sin = torch.cos(ang), torch.sin(ang)
    xf, yf = ((rb - lt) / 2).split(1, 1)
    xy = torch.cat([xf * cos - yf * sin, xf * sin + yf * cos], 1) + anc
    box = torch.cat([xy, lt + rb], 1) * st
    scores = logits.sigmoid()
    k = min(max_det, scores.shape[0])
    best = scores.max(1).values.topk(k).indices
    sc = scores[best].flatten()
    top = sc.topk(k)
    cand = best[top.indices // nc]
    cls = top.indices % nc
    out = []
    for v, i, c in zip(top.values.tolist(), cand.tolist(), cls.tolist()):
        if v < conf:
            break
        out.append((*box[i].tolist(), float(ang[i]), v, int(c)))
    return out


def corners(cx, cy, w, h, a):
    c, s = math.cos(a), math.sin(a)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return pts
