"""Helpers shared by the Strassen probes in this folder. Importing it touches no hardware: the
board helpers import the engine modules only when called."""
import os
import sys

import numpy as np
import torch

BF, FP = torch.bfloat16, torch.float16
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # models/alpamayo2_edge
# a fused activation: the matmat_mul_core arguments, and what the hardware computes (its GELU is x * sigmoid(1.702 x))
ACT_KW = {"gelu": {"gelu_enable": True}, "silu": {"silu_enable": True},
          "relu": {"clamp_enable": True, "clamp_min": 0.0, "clamp_max": float("inf")}, None: {}}
ACT_FN = {"gelu": lambda v: v * torch.sigmoid(1.702 * v), "silu": lambda v: v * torch.sigmoid(v),
          "relu": lambda v: v.clamp(min=0), None: lambda v: v}


def snr(ref, x):
    ref, x = ref.double(), x.double()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-300)))


def decode_if4(data, scales, N, K):
    """An IF4 pack -> code x |scale| in float64, no rounding: the sign of a block's scale says
    whether its 64 codes are integers -8..7 or the small-float table. (quant_lib.dequant multiplies
    in bf16 and is off by up to a bf16 step, so it cannot check an fp16 result.)"""
    nib = np.frombuffer(data, dtype=np.uint8).reshape(N, K // 2)
    codes = np.stack([nib & 0xF, nib >> 4], -1).reshape(N, K // 64, 64)
    sc = torch.frombuffer(bytearray(scales), dtype=torch.int16).view(BF).double().numpy().reshape(N, K // 64, 1)
    table = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    ints = np.where(codes >= 8, codes.astype(np.int64) - 16, codes).astype(np.float64)
    return torch.from_numpy((np.where(sc < 0, ints, table[codes]) * np.abs(sc)).reshape(N, K))


# ---- Strassen on quarters, for the host models and the block-layout board runs
QUARTERS = ((0, 0), (0, 1), (1, 0), (1, 1))
# 7 products = (input sum) . (weight sum)^T. Input quarter index = 2 * row half + K half; weight
# quarter index = 2 * N half + K half (T_DEF is nn_lib._STRASSEN_W). An entry: (quarter, other
# quarter or None, sign).
S_DEF = [(0, 3, +1), (2, 3, +1), (0, None, 0), (3, None, 0), (0, 1, +1), (2, 0, -1), (1, 3, -1)]
T_DEF = [(0, 3, +1), (0, None, 0), (2, 3, -1), (1, 0, -1), (3, None, 0), (0, 2, +1), (1, 3, +1)]
# C quarter (2 * row half + N half) from the products: signed sums, left to right
C_DEF = [[(0, +1), (3, +1), (4, -1), (6, +1)], [(2, +1), (4, +1)], [(1, +1), (3, +1)], [(0, +1), (1, -1), (2, +1), (5, +1)]]


def quarters(t):
    a, b = t.shape[0] // 2, t.shape[1] // 2
    return [t[i * a:(i + 1) * a, j * b:(j + 1) * b] for i, j in QUARTERS]


def combo(q, a, b, sign):
    return q[a] if b is None else q[a] + sign * q[b]


def to_blocks(t, L):
    """Recursive block layout: a matrix = its 4 quarters one after another, each laid out the same way."""
    return t.reshape(-1) if L == 0 else torch.cat([to_blocks(q, L - 1) for q in quarters(t)])


def from_blocks(v, rows, cols, L):
    if L == 0:
        return v.reshape(rows, cols)
    q = [from_blocks(c, rows // 2, cols // 2, L - 1) for c in v.chunk(4)]
    return torch.cat([torch.cat(q[:2], 1), torch.cat(q[2:], 1)])


def leaves(W, L, plain=True):
    """The level-L leaf weights in product order: [(exact sum of blocks of W, plain?)].
    plain: no sum on the path to this block, i.e. it is a block of W itself."""
    if L == 0:
        return [(W, plain)]
    wq = quarters(W)
    return [x for t in T_DEF for x in leaves(combo(wq, *t), L - 1, plain and t[1] is None)]


def products(X, leaf, L, top=True):
    """Strassen level L on the host with the leaf weights from the iterator `leaf` (product
    order): input sums, products and recombination rounded to fp16, the top result to bf16."""
    r = lambda t: t.to(FP).double()
    if L == 0:
        return r(X @ next(leaf).T)
    xq = quarters(X)
    m = [products(r(combo(xq, *s)) if s[1] is not None else xq[s[0]], leaf, L - 1, False) for s in S_DEF]
    out = []
    for terms in C_DEF:
        acc = m[terms[0][0]]
        for i, (k, sign) in enumerate(terms[1:]):
            acc = acc + sign * m[k]
            acc = acc.to(BF).double() if (top and i == len(terms) - 2) else r(acc)
        out.append(acc)
    return torch.cat([torch.cat(out[:2], 1), torch.cat(out[2:], 1)])


# ---- the board
def open_board(dev=None, engines=1):
    """-> (the alpamayo2_edge_test module, an engine, engines used, cores the bitstream reports).
    engines: a count or 'max'."""
    sys.path.insert(0, HERE)
    import alpamayo2_edge_test as T
    import user_dma_core
    dev = dev or T.CFG["defaults"]["dev"]
    user_dma_core.set_dma_device(dev)
    user_dma_core.configure_clock_from_hardware()
    cores = getattr(user_dma_core, "ANDROMEDA_CORE_COUNT", None) or 1
    NE = cores if engines == "max" else int(engines)
    assert 1 <= NE <= cores, f"--engines {NE} but the bitstream reports {cores} core(s)"
    return T, T.Alpamayo2Edge_UnifiedEngine(dev=dev, engines=NE), NE, cores


def put(ue, t):
    """Upload a bf16 or fp16 tensor as it is; -> its address."""
    a = ue._alloc(t.numel())
    ue.dma_to_accelerator_memory(a, t.contiguous().flatten())
    return a


def run_regs(ue, emit, timeout=600.0):
    """One transient program: emit(reg), reg(v) giving a register that holds v.
    -> (board us, instructions emitted)."""
    regs, count = {}, [0]

    def reg(v):
        if v not in regs:
            regs[v] = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(regs[v], v)
        return regs[v]

    def body():
        ue.reset_isa_reg_counter()
        emit(reg)
        count[0] = ue.capture_count
    us = ue._run_transient(body, timeout)
    return us, count[0]


def hangs(M, K, N):
    """The softmax-free bf16 / fp16 matmul shape that took the board down twice (2026-11-18): a
    short K whose rows do not fit one on-chip tile."""
    from user_dma_core import URAM_FULL_ELEMENTS
    return N * (K // 64) <= 4095 and M * (K + -(-N // 64) * 64) > URAM_FULL_ELEMENTS


def block_strassen(ue, reg, leaf, x, xf, t, out, of, m, k, n, lv):
    """Emit Strassen on operands in block layout: x [m, k] (format xf) -> out [m, n] (format of).
    t: the leaf weights as nested lists of 7 (T_DEF order); leaf(reg, m, k, n, x, w, out, xf, of)
    emits a leaf matmul. Sums, products and recombination are fp16; scratch from ue._alloc."""
    from storage_format import StorageFormat, TensorFormats
    from user_dma_core import UE_MODE
    F16 = StorageFormat.FP16
    if lv == 0:
        leaf(reg, m, k, n, x, t, out, xf, of)
        return
    h, kk, nn = m // 2, k // 2, n // 2
    elt = lambda a, c, o, sign, cols, fm: ue.eltwise_core_dram(
        M=h, N=cols, dram_a=a, dram_b=c, dram_out=o,
        mode=UE_MODE.ELTWISE_ADD if sign > 0 else UE_MODE.ELTWISE_SUB, gpr_M_reg=reg(h), formats=fm)
    xq = [x + i * h * kk * 2 for i in range(4)]
    prod = []
    for s, child in zip(S_DEF, t):
        if s[1] is None:
            src, sf = xq[s[0]], xf
        else:
            src, sf = ue._alloc(h * kk), F16
            elt(xq[s[0]], xq[s[1]], src, s[2], kk, TensorFormats(a=xf, b=xf, output=F16))
        p = ue._alloc(h * nn)
        block_strassen(ue, reg, leaf, src, sf, child, p, F16, h, kk, nn, lv - 1)
        prod.append(p)
    tmp = [ue._alloc(h * nn) for _ in range(2)]
    for qi, terms in enumerate(C_DEF):
        acc = prod[terms[0][0]]
        for i, (kx, sign) in enumerate(terms[1:]):
            last = i == len(terms) - 2
            dst = out + qi * h * nn * 2 if last else tmp[i % 2]
            elt(acc, prod[kx], dst, sign, nn, TensorFormats(a=F16, b=F16, output=of if last else F16))
            acc = dst
