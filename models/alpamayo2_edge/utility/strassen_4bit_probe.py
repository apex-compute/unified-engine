#!/usr/bin/env python3
"""Does Strassen work on a 4-bit (IF4) matmul, and what do fp16 sums buy there?

The prefill matmul shapes, Strassen level 1, 2, 3 with the input sums, the products and the
recombination in fp16 (as in nn_lib.strassen_matmul_core_dram), and these ways to hold the weights:

  a  native bf16 matmul of the unquantized weight                       <- the reference
  b  native IF4 matmul                                                  <- what the model runs today
  c  Strassen, weight sums of the UNQUANTIZED weight, each leaf re-quantized to IF4
  d  Strassen, weight sums of the 4-BIT (dequantized IF4) weight, leaves in fp16
  e  Strassen, weight sums of the unquantized weight, leaves in fp16                  (16-bit weights)
  f1 Strassen, weight sums of the unquantized weight, leaves re-quantized to IF8
  f2 Strassen, weight sums of the 4-bit weight, leaves re-quantized to IF8
  g1 hybrid: a leaf that is a plain quarter of the 4-bit weight (2 of 7 per level: no sum on its
     path) stays IF4 as it is; the true sums of the 4-bit weight go to fp16
  g2 the same hybrid with the true sums in IF8

Per shape and level: SNR against (a), SNR against (b) -- that one says whether the 4-bit model's
own behaviour is kept -- and the weight bytes relative to the plain IF4 weight (IF4 0.5 byte per
weight + one bf16 scale per 64, IF8 1 byte + scale, fp16 2 bytes; times the tree, (7/4)^level).

  python models/alpamayo2_edge/utility/strassen_4bit_probe.py --host           # numeric model, no board (a 512 x 512 weight)
  python models/alpamayo2_edge/utility/strassen_4bit_probe.py --host --real    # on corners of real Alpamayo LM weights
  python models/alpamayo2_edge/utility/strassen_4bit_probe.py --host --crop 0  # the whole prefill shapes (minutes)
  python models/alpamayo2_edge/utility/strassen_4bit_probe.py                  # the board, one engine

--host rounds to fp16 / bf16 wherever the board stores a value and quantizes with quant_lib. The
same model in strassen_fp16_probe.py matched the board within ~1.5 dB for fp16 leaves and to
0.1 dB for IF4 leaves.
"""
import argparse
import json
import os
import sys

import torch

BF, FP = torch.bfloat16, torch.float16
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # repo root: quant_lib
import quant_lib  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
QUARTERS = ((0, 0), (0, 1), (1, 0), (1, 1))
# 7 products = (input sum) . (weight sum)^T. Input quarter index = 2 * row half + K half; weight
# quarter index = 2 * N half + K half. An entry: (quarter, other quarter or None, sign).
S_DEF = [(0, 3, +1), (2, 3, +1), (0, None, 0), (3, None, 0), (0, 1, +1), (2, 0, -1), (1, 3, -1)]
T_DEF = [(0, 3, +1), (0, None, 0), (2, 3, -1), (1, 0, -1), (3, None, 0), (0, 2, +1), (1, 3, +1)]
C_DEF = [[(0, +1), (3, +1), (4, -1), (6, +1)], [(2, +1), (4, +1)], [(1, +1), (3, +1)], [(0, +1), (1, -1), (2, +1), (5, +1)]]
SHAPES = [("q / o", 408, 2048, 2048), ("k / v", 408, 2048, 1024), ("fc1 (one half)", 408, 2048, 4608), ("fc2 (one half)", 408, 4608, 2048)]
# key, label, source of the weight sums ("raw" unquantized | "q4" the dequantized IF4 weight),
# storage of a true sum, storage of a plain quarter ("keep" = the IF4 values as they are)
VARIANTS = [("c", "IF4 leaves, sums of the unquantized weight", "raw", "if4", "if4"),
            ("d", "fp16 leaves, sums of the 4-bit weight", "q4", "fp16", "fp16"),
            ("e", "fp16 leaves, sums of the unquantized weight", "raw", "fp16", "fp16"),
            ("f1", "IF8 leaves, sums of the unquantized weight", "raw", "if8", "if8"),
            ("f2", "IF8 leaves, sums of the 4-bit weight", "q4", "if8", "if8"),
            ("g1", "hybrid: quarters stay IF4, sums in fp16", "q4", "fp16", "keep"),
            ("g2", "hybrid: quarters stay IF4, sums in IF8", "q4", "if8", "keep")]
BYTES = {"if4": 0.5 + 2 / 64, "keep": 0.5 + 2 / 64, "if8": 1 + 2 / 64, "fp16": 2.0}


def size_factor(L, sums, plain):
    """Weight bytes of the level-L tree over the plain IF4 weight. 2^L of the 7^L leaves are plain quarters."""
    return (2 ** L * BYTES[plain] + (7 ** L - 2 ** L) * BYTES[sums]) / 4 ** L / BYTES["if4"]


def snr(ref, x):
    ref, x = ref.double(), x.double()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-300)))


def quarters(t):
    a, b = t.shape[0] // 2, t.shape[1] // 2
    return [t[i * a:(i + 1) * a, j * b:(j + 1) * b] for i, j in QUARTERS]


def combo(q, a, b, sign):
    return q[a] if b is None else q[a] + sign * q[b]


def deq(prec, W):
    """What a weight is once it is stored in `prec` (if4 / if8)."""
    Wc = W.float().cpu().contiguous()
    data, scale = quant_lib.quantize(prec, Wc)
    return quant_lib.dequant(prec, data, scale, *Wc.shape).double().to(W.device)


def leaves(W, L, plain=True):
    """The level-L leaf weights in product order: [(exact fp64 sum of the source weight, plain?)].
    plain: no sum on the path to this block, i.e. it is a quarter of the source itself."""
    if L == 0:
        return [(W, plain)]
    wq = quarters(W)
    return [x for t in T_DEF for x in leaves(combo(wq, *t), L - 1, plain and t[1] is None)]


def stored(blocks, how):
    """What the blocks are once stored as `how`. All of one call are quantized as ONE stacked
    matrix (rows are independent in IF4 / IF8, and a quantize call has a large fixed cost)."""
    if how == "keep" or not blocks:
        return list(blocks)
    if how == "fp16":
        return [b.to(FP).double() for b in blocks]
    return list(deq(how, torch.cat(blocks)).split([b.shape[0] for b in blocks]))


def host_model(X, W, L, sums, plain_how):
    """Strassen level L: input sums / products / recombination rounded to fp16, the top result to
    bf16; true weight sums stored as `sums`, plain quarters as `plain_how`."""
    lv = leaves(W, L)
    val = [None] * len(lv)
    for how, want in ((plain_how, True), (sums, False)):
        idx = [i for i, (_, p) in enumerate(lv) if p == want]
        for i, v in zip(idx, stored([lv[i][0] for i in idx], how)):
            val[i] = v
    return _products(X, iter(val), L, True)


def _products(X, leaf, L, top):
    r = lambda t: t.to(FP).double()
    if L == 0:
        return r(X @ next(leaf).T)
    xq = quarters(X)
    m = [_products(r(combo(xq, *s)) if s[1] is not None else xq[s[0]], leaf, L - 1, False) for s in S_DEF]
    out = []
    for terms in C_DEF:
        acc = m[terms[0][0]]
        for i, (k, sign) in enumerate(terms[1:]):
            acc = acc + sign * m[k]
            acc = acc.to(BF).double() if (top and i == len(terms) - 2) else r(acc)
        out.append(acc)
    return torch.cat([torch.cat(out[:2], 1), torch.cat(out[2:], 1)])


def real_weights(layers):
    """[(name, W [N, K] fp32)] from the Alpamayo checkpoint (host only; nothing here touches hardware)."""
    sys.path.insert(0, HERE)
    from alpamayo2_edge_cpu_reference import LM, resolve_model_dir
    from safetensors import safe_open
    d = resolve_model_dir(None)
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]

    def get(name):
        with safe_open(str(d / index[name]), framework="pt") as f:
            return f.get_tensor(name).float()

    out = []
    for li in layers:
        p = f"{LM}layers.{li}."
        fc1, fc2 = get(p + "mlp.fc1.weight"), get(p + "mlp.fc2.weight")
        half = fc1.shape[0] // 2
        out += [(f"layer {li} q_proj", get(p + "self_attn.q_proj.weight")), (f"layer {li} k_proj", get(p + "self_attn.k_proj.weight")),
                (f"layer {li} o_proj", get(p + "self_attn.o_proj.weight")), (f"layer {li} fc1 (one half)", fc1[:half].contiguous()),
                (f"layer {li} fc2 (one half)", fc2[:, :half].contiguous())]
    return out


def host(args):
    g = torch.Generator().manual_seed(args.seed)
    c = args.crop
    if args.real:
        cases = [(name, W) for name, W in real_weights(args.layers)]
    elif c:
        cases = [("random weight", torch.randn(c, c, generator=g) / c ** 0.5)]
    else:
        cases = [(f"{name}, random weight", torch.randn(N, K, generator=g) / K ** 0.5) for name, _, K, N in SHAPES]
    for name, W in cases:
        if c:                                                               # light: the top-left corner (whole 64-blocks, real statistics)
            W = W[:c, :c].contiguous()
            name += f", {c} x {c} corner" if args.real else ""
        N, K = W.shape
        M = args.rows
        X = torch.randn(M, K, generator=g).to(BF).double().to(DEV)
        W = W.double().to(DEV)
        Wq4 = deq("if4", W)
        a = (X @ W.to(BF).double().T).to(BF).double()                       # native bf16
        b = (X @ Wq4.T).to(BF).double()                                     # native IF4
        print(f"\n{name}: X [{M}, {K}] . W [{N}, {K}]^T")
        print(f"  {'':54s} {'vs bf16':>8} {'vs IF4':>8} {'weights':>8}")
        print(f"  {'b  native IF4':54s} {snr(a, b):5.1f} dB {'':>8} {1.0:7.2f}x", flush=True)
        for L in range(1, args.levels + 1):
            if K % (64 << L) or N % (64 << L) or M % (1 << L):
                print(f"  level {L}: the shape does not halve {L} times, skipped")
                continue
            for key, label, src, sums, plain in VARIANTS:
                C = host_model(X, W if src == "raw" else Wq4, L, sums, plain)
                print(f"  {key:2s} L{L} {label:48s} {snr(a, C):5.1f} dB {snr(b, C):5.1f} dB {size_factor(L, sums, plain):7.2f}x", flush=True)


def board(args):
    """Every variant on the device, one engine: SNR against the board's native bf16 result and
    against its native IF4 result, time against native IF4."""
    sys.path.insert(0, HERE)
    import alpamayo2_edge_test as T
    import user_dma_core
    from storage_format import StorageFormat, TensorFormats
    from user_dma_core import TYPE, UE_MODE, URAM_FULL_ELEMENTS
    dev = args.dev or T.CFG["defaults"]["dev"]
    user_dma_core.set_dma_device(dev)
    user_dma_core.configure_clock_from_hardware()
    ue = T.Alpamayo2Edge_UnifiedEngine(dev=dev, engines=1)
    base = ue.get_tensor_dram_addr()
    F16, B16 = StorageFormat.FP16, StorageFormat.BF16
    QT = {"if4": TYPE.IF4, "if8": TYPE.IF8}

    def put(t):
        a = ue._alloc(t.numel())
        ue.dma_to_accelerator_memory(a, t.contiguous().flatten())
        return a

    def pack(prec, W):                                                      # -> (type, data address, scale address)
        Wc = W.float().cpu().contiguous()
        data, scale = quant_lib.quantize(prec, Wc)
        return (prec, *ue._qstore((data, scale, *Wc.shape)))

    def store(W, how):
        """keep: a quarter of the 4-bit weight, packed again as IF4 (its blocks of 64 are whole, so
        the pack holds the same values)."""
        return put(W.to(FP)) if how == "fp16" else pack("if4" if how == "keep" else how, W)

    def run(emit):
        regs = {}

        def reg(v):
            if v not in regs:
                regs[v] = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(regs[v], v)
            return regs[v]

        def body():
            ue.reset_isa_reg_counter()
            emit(reg)
        return ue._run_transient(body, 600.0)

    def hangs(M, K, N):
        """The softmax-free bf16 / fp16 matmul shape that took the board down twice (2026-11-18)."""
        return N * (K // 64) <= 4095 and M * (K + -(-N // 64) * 64) > URAM_FULL_ELEMENTS

    def matmul(reg, m, k, n, a, w, out, fa, fo):
        """w: an fp16 / bf16 weight address (bf16 only for the native), or (type, data, scales)."""
        if isinstance(w, tuple):
            ue.matmat_mul_core(M=m, K=k, N=n, A_DRAM_ADDR=a, B_DRAM_ADDR=w[1], OUTPUT_DRAM_ADDR=out, gpr_M_reg=reg(m),
                               is_B_quantized=True, data_type=QT[w[0]], SCALE_DRAM_ADDR=w[2], formats=TensorFormats(a=fa, output=fo))
        else:
            ue.matmat_mul_core(M=m, K=k, N=n, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out, gpr_M_reg=reg(m),
                               formats=TensorFormats(a=fa, b=F16, output=fo))

    def to_blocks(t, L):
        return t.reshape(-1) if L == 0 else torch.cat([to_blocks(q, L - 1) for q in quarters(t)])

    def from_blocks(v, rows, cols, L):
        if L == 0:
            return v.reshape(rows, cols)
        q = [from_blocks(c, rows // 2, cols // 2, L - 1) for c in v.chunk(4)]
        return torch.cat([torch.cat(q[:2], 1), torch.cat(q[2:], 1)])

    g = torch.Generator().manual_seed(args.seed)
    for name, M, K, N in SHAPES:
        ue._tensor_dram_addr = base
        W = torch.randn(N, K, generator=g) / K ** 0.5
        X = torch.randn(M, K, generator=g).to(BF)
        Wq4 = deq("if4", W).float()
        a_x, a_c = put(X), ue._alloc(M * N)
        a_wb, w4 = put(W.to(BF)), pack("if4", W)
        assert not hangs(M, K, N)
        t_bf = run(lambda reg: ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a_x, B_DRAM_ADDR=a_wb, OUTPUT_DRAM_ADDR=a_c, gpr_M_reg=reg(M)))
        a = ue.dma_from_accelerator_memory(a_c, (M, N)).double()
        t_if4 = run(lambda reg: matmul(reg, M, K, N, a_x, w4, a_c, B16, B16))
        b = ue.dma_from_accelerator_memory(a_c, (M, N)).double()
        print(f"\n{name}: X [{M}, {K}] . W [{N}, {K}]^T, random weight, one engine")
        print(f"  {'':54s} {'vs bf16':>8} {'vs IF4':>8} {'weights':>8} {'board ms':>9} {'time':>6}")
        print(f"  {'a  native bf16':54s} {'':>8} {'':>8} {2.0 / BYTES['if4']:7.2f}x {t_bf / 1e3:9.2f} {t_bf / t_if4:5.2f}x")
        print(f"  {'b  native IF4':54s} {snr(a, b):5.1f} dB {'':>8} {1.0:7.2f}x {t_if4 / 1e3:9.2f} {1.0:5.2f}x", flush=True)
        mark = ue.get_tensor_dram_addr()
        for L in range(1, args.levels + 1):
            if K % (64 << L) or N % (64 << L) or M % (1 << L):
                continue
            for key, label, src, sums, plain in VARIANTS:
                tag = f"{key:2s} L{L} {label}"
                if "fp16" in (sums, plain) and hangs(M >> L, K >> L, N >> L):
                    print(f"  {tag}: its fp16 leaf matmul is the shape that hangs the board, skipped")
                    continue
                ue._tensor_dram_addr = mark

                def tree(Wn, lv, is_plain):                                 # weight sums in fp32 on the host
                    if lv == 0:
                        return store(Wn, plain if is_plain else sums)
                    wq = quarters(Wn)
                    return [tree(combo(wq, *t), lv - 1, is_plain and t[1] is None) for t in T_DEF]

                a_xz, a_t, a_out = put(to_blocks(X, L)), tree(W if src == "raw" else Wq4, L, True), ue._alloc(M * N)

                def node(reg, x, xf, t, out, of, m, k, n, lv):
                    if lv == 0:
                        matmul(reg, m, k, n, x, t, out, xf, of)
                        return
                    h, kk, nn = m // 2, k // 2, n // 2
                    elt = lambda p, c, o, sign, cols, fm: ue.eltwise_core_dram(
                        M=h, N=cols, dram_a=p, dram_b=c, dram_out=o,
                        mode=UE_MODE.ELTWISE_ADD if sign > 0 else UE_MODE.ELTWISE_SUB, gpr_M_reg=reg(h), formats=fm)
                    xq = [x + i * h * kk * 2 for i in range(4)]
                    prod = []
                    for s, child in zip(S_DEF, t):
                        if s[1] is None:
                            src_a, sf = xq[s[0]], xf
                        else:
                            src_a, sf = ue._alloc(h * kk), F16
                            elt(xq[s[0]], xq[s[1]], src_a, s[2], kk, TensorFormats(a=xf, b=xf, output=F16))
                        p = ue._alloc(h * nn)
                        node(reg, src_a, sf, child, p, F16, h, kk, nn, lv - 1)
                        prod.append(p)
                    tmp = [ue._alloc(h * nn) for _ in range(2)]
                    for qi, terms in enumerate(C_DEF):
                        acc = prod[terms[0][0]]
                        for i, (kx, sign) in enumerate(terms[1:]):
                            last = i == len(terms) - 2
                            dst = out + qi * h * nn * 2 if last else tmp[i % 2]
                            elt(acc, prod[kx], dst, sign, nn, TensorFormats(a=F16, b=F16, output=of if last else F16))
                            acc = dst

                us = run(lambda reg: node(reg, a_xz, B16, a_t, a_out, B16, M, K, N, L))
                C = from_blocks(ue.dma_from_accelerator_memory(a_out, (M * N,)).double(), M, N, L)
                print(f"  {tag:54s} {snr(a, C):5.1f} dB {snr(b, C):5.1f} dB {size_factor(L, sums, plain):7.2f}x {us / 1e3:9.2f} {us / t_if4:5.2f}x"
                      + ("" if bool(torch.isfinite(C).all()) else "  NaN/Inf!"), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="store_true", help="numeric model only, no board")
    ap.add_argument("--real", action="store_true", help="--host on real Alpamayo LM weights (q, k, o, fc1 half, fc2 half)")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 14], help="LM layers for --real")
    ap.add_argument("--rows", type=int, default=128, help="rows of X in --host")
    ap.add_argument("--crop", type=int, default=512, help="--host: use only this corner of every weight (fast); 0 = the whole prefill shapes")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert not args.real or args.host, "--real is a host mode: add --host"
    host(args) if args.host else board(args)


if __name__ == "__main__":
    main()
