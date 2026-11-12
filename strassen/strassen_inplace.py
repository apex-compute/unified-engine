"""Drop-in Strassen: ordinary row-major [M, K] in, ordinary row-major [M, N] out, bias included.
This is the form a model can use in place of one matmat_mul_core, and the probe times ALL of it
(the earlier probes left the input split and the output in blocks).

    C = X . W^T + b      default 720 x 4352 -> 1152 (the Alpamayo / SigLIP fc2 shape), bf16, one engine

How the output lands in place without a stitching pass: the LAST product of each output quarter
adds the rest of that quarter as a full-matrix bias and writes straight into the [M, N] buffer
at the row pitch (matmat_mul_core's output-stride path).

    gather  X11 X12 X21 X22                  4 strided reads of the input (h x k each)
    sums    S1..S5                           5 eltwise
    P1' = (X11+X22)(Y11+Y22) + (b_L + b_R)   products with the layer bias folded in, so that
    P2' = (X21+X22) Y11      +  b_L          U below is bias-free and every quarter ends up
    P5' = (X11+X12) Y22      +  b_R          with exactly its own half of b
    C12 = X11 (Y12-Y22)      + P5'    ->  written in place (top rows, right half)
    C21 = X22 (Y21-Y11)      + P2'    ->  written in place (bottom rows, left half)
    U   = P1' - P2' - P5'                    2 eltwise
    C22 = (X21-X11)(Y11+Y12) + (U + C12)  -> in place     (C12 / C21 are read back with one
    C11 = (X12-X22)(Y21+Y22) + (U + C21)  -> in place      strided gather each; 2 more eltwise)

7 products, 9 eltwise, 6 gathers -- against one native matmul with the same bias.

Run from the repo root:
    python strassen/strassen_inplace.py                # the board
    python strassen/strassen_inplace.py --selfcheck    # host only: checks the algebra and layout
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from strassen_levels import BPE, Device, bf16, cos_sim, say, snr_db  # noqa: E402  (shared harness; no device at import)

GATHER_ELEMENTS = 131072           # elements per strided SRAM hop (the size the model gathers use)


def plan(M, K, N, base):
    """Addresses of every tensor buffer, bump-allocated from `base`. Returns (dict, bytes)."""
    h, k, n = M // 2, K // 2, N // 2
    a, t = {}, base
    for name, elements in ([("x", M * K), ("out", M * N), ("native", M * N)] + [(f"xq{i}", h * k) for i in range(4)]
                           + [(f"s{i}", h * k) for i in range(5)]
                           + [(name, h * n) for name in ("p1", "p2", "p5", "u", "t1", "t2", "c12", "c21")]):
        a[name], t = t, t + elements * BPE
    return a, t - base


def weight_plan(K, N, base):
    k, n = K // 2, N // 2
    a, t = {}, base
    for name, elements in [("w", N * K)] + [(f"c{i}", n * k) for i in range(7)] + [("b", N), ("b_l", n), ("b_r", n), ("b_sum", n)]:
        a[name], t = t, t + elements * BPE
    return a, t - base


def host_weights(W, b):
    """name -> tensor for every weight-side buffer. Yij = block (i, j) of W^T."""
    n, k = W.shape[0] // 2, W.shape[1] // 2
    Y11, Y12, Y21, Y22 = W[:n, :k], W[n:, :k], W[:n, k:], W[n:, k:]
    out = {"w": W, "b": b, "b_l": b[:n], "b_r": b[n:], "b_sum": b[:n] + b[n:]}
    for i, combo in enumerate((Y11 + Y22, Y11, Y12 - Y22, Y21 - Y11, Y22, Y11 + Y12, Y21 + Y22)):
        out[f"c{i}"] = combo
    return out


def emit_native(be, M, K, N, t, w):
    be.mm(M, K, N, t["x"], w["w"], t["native"], bias=w["b"])


def emit_strassen(be, M, K, N, t, w, parts=("gather", "sums", "products")):
    """parts narrows the program for the timing breakdown (results are only valid with all three)."""
    h, k, n = M // 2, K // 2, N // 2
    x, out = t["x"], t["out"]
    xq = [t[f"xq{i}"] for i in range(4)]                # X11, X12, X21, X22
    s = [t[f"s{i}"] for i in range(5)]
    if "gather" in parts:
        for i, (r0, c0) in enumerate(((0, 0), (0, k), (h, 0), (h, k))):
            be.gather(x + (r0 * K + c0) * BPE, K, k, h, xq[i])
    x11, x12, x21, x22 = xq
    if "sums" in parts:
        for (a, b, mode), o in zip([(x11, x22, "add"), (x21, x22, "add"), (x11, x12, "add"), (x21, x11, "sub"), (x12, x22, "sub")], s):
            be.elt(h, k, a, b, o, mode)
    P = "products" in parts
    if P:
        be.mm(h, k, n, s[0], w["c0"], t["p1"], bias=w["b_sum"])
        be.mm(h, k, n, s[1], w["c1"], t["p2"], bias=w["b_l"])
        be.mm(h, k, n, s[2], w["c4"], t["p5"], bias=w["b_r"])
        be.mm(h, k, n, x11, w["c2"], out + n * BPE, full_bias=t["p5"], stride=N)                # C12
        be.mm(h, k, n, x22, w["c3"], out + h * N * BPE, full_bias=t["p2"], stride=N)            # C21
    if "gather" in parts:
        be.gather(out + n * BPE, N, n, h, t["c12"])
        be.gather(out + h * N * BPE, N, n, h, t["c21"])
    if "sums" in parts:
        be.elt(h, n, t["p1"], t["p2"], t["u"], "sub")
        be.elt(h, n, t["u"], t["p5"], t["u"], "sub")
        be.elt(h, n, t["u"], t["c12"], t["t2"], "add")
        be.elt(h, n, t["u"], t["c21"], t["t1"], "add")
    if P:
        be.mm(h, k, n, s[3], w["c5"], out + (h * N + n) * BPE, full_bias=t["t2"], stride=N)     # C22
        be.mm(h, k, n, s[4], w["c6"], out, full_bias=t["t1"], stride=N)                         # C11


class HostModel:
    """The device ops on one flat fp32 array, rounding to bf16 on every write."""

    def __init__(self, elements):
        self.mem = torch.zeros(elements)

    def wr(self, addr, t):
        t = bf16(t.reshape(-1))
        self.mem[addr // BPE: addr // BPE + t.numel()] = t

    def rd(self, addr, n):
        return self.mem[addr // BPE: addr // BPE + n]

    def rows(self, addr, pitch, cols, rows):
        idx = (addr // BPE + torch.arange(rows)[:, None] * pitch + torch.arange(cols)[None, :]).reshape(-1)
        return idx

    def mm(self, m, k, n, a, w, out, bias=None, full_bias=None, stride=None):
        y = self.rd(a, m * k).reshape(m, k) @ self.rd(w, n * k).reshape(n, k).T
        if bias is not None:
            y = y + self.rd(bias, n)
        if full_bias is not None:
            y = y + self.rd(full_bias, m * n).reshape(m, n)
        self.mem[self.rows(out, stride or n, n, m)] = bf16(y.reshape(-1))

    def elt(self, rows, cols, a, b, out, mode):
        x, y = self.rd(a, rows * cols), self.rd(b, rows * cols)
        self.wr(out, x + y if mode == "add" else x - y)

    def gather(self, src, pitch, cols, rows, dst):
        self.wr(dst, self.mem[self.rows(src, pitch, cols, rows)])


class Board(Device):
    """strassen_levels.Device plus bias / output-stride matmuls and the strided gather."""

    def mm(self, m, k, n, a, w, out, bias=None, full_bias=None, stride=None):
        kw = dict(C_DRAM_ADDR=bias, bias_mode="broadcast_N")
        if full_bias is not None:
            kw = dict(C_DRAM_ADDR=full_bias, bias_mode="full_matrix")
        if stride is not None:
            kw["gpr_out_row_stride_reg"] = self.regs[("stride", stride)]
        self.ue.matmat_mul_core(M=m, K=k, N=n, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out,
                                gpr_M_reg=self.regs[m], **kw)

    def gather(self, src, pitch, cols, rows, dst):
        step = max(1, GATHER_ELEMENTS // cols)
        for r0 in range(0, rows, step):
            take = min(step, rows - r0)
            stride = dict(stride_bytes_per_chunk=cols * BPE, stride_jump_bytes=pitch * BPE) if take > 1 else {}
            self.ue.accelerator_memory_to_sram(accelerator_dram_address=src + r0 * pitch * BPE, sram_address=0x00000,
                                               element_size=take * cols, **stride)
            self.ue.sram_to_accelerator_memory(sram_address=0x00000, accelerator_dram_address=dst + r0 * cols * BPE,
                                               element_size=take * cols)

    def compile_with(self, M, N, body):
        """Row-count GPRs for M and M/2, plus the output row pitch."""
        global_regs = {}

        def wrapped():
            r = self.ue.alloc_isa_reg()
            self.ue.generate_instruction_add_set(r, N)
            self.regs[("stride", N)] = r
            global_regs.update(self.regs)
            body()
        return self.compile([M, M // 2], wrapped)


def data(M, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    X = bf16(torch.randn(M, K, generator=g))
    W = bf16(torch.randn(N, K, generator=g) / K ** 0.5)
    b = bf16(torch.randn(N, generator=g) * 0.1)
    return X, W, b, X.double() @ W.double().T + b.double()


def selfcheck(args):
    M, K, N = 48, 512, 384
    X, W, b, ref = data(M, K, N, args.seed)
    t, t_bytes = plan(M, K, N, 0)
    w, w_bytes = weight_plan(K, N, t_bytes)
    hm = HostModel((t_bytes + w_bytes) // BPE)
    for name, val in host_weights(W, b).items():
        hm.wr(w[name], val)
    hm.wr(t["x"], X)
    emit_native(hm, M, K, N, t, w)
    emit_strassen(hm, M, K, N, t, w)
    nat, st = hm.rd(t["native"], M * N).reshape(M, N), hm.rd(t["out"], M * N).reshape(M, N)
    say(f"host bf16 model, [{M}, {K}] -> {N}: native {snr_db(ref, nat):.1f} dB, drop-in strassen {snr_db(ref, st):.1f} dB "
        f"(cos {cos_sim(ref, st):.6f})")
    ok = snr_db(ref, st) > 35
    say("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--M", type=int, default=720)
    ap.add_argument("--K", type=int, default=4352)
    ap.add_argument("--N", type=int, default=1152)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true", help="host only")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args)
    M, K, N = args.M, args.K, args.N
    assert M % 2 == 0 and K % 128 == 0 and N % 128 == 0, "M must be even; K and N multiples of 128"
    X, W, b, ref = data(M, K, N, args.seed)

    d = Board(args.dev)
    t, t_bytes = plan(M, K, N, d.tbase)
    w, w_bytes = weight_plan(K, N, d.wbase)
    assert w_bytes <= d.w_room and t_bytes <= d.t_room, "shape does not fit the default DRAM map"
    d.zero(d.tbase, t_bytes // BPE)
    for name, val in host_weights(W, b).items():
        d.upload(w[name], val)
    d.upload(t["x"], X)
    say(f"shape [{M}, {K}] x [{K} -> {N}] + bias, bf16, one engine (peak {d.peak:.1f} GFLOPS); row-major in, row-major out\n")

    limit = 30.0 + 2 * M * K * N / 2e9
    timed = {}

    def run(name, body):
        d.compile_with(M, N, body)
        say(f"  {name}..", end="\r")
        try:
            timed[name] = min(d.run(limit) for _ in range(args.runs)) / 1e3
        except Exception as exc:
            say(f"  {name}: FAILED {type(exc).__name__}: {str(exc)[:110]}")
            say("stopping: a timed-out program leaves the engine stuck; reset the board before the next run")
            sys.exit(1)

    run("native", lambda: emit_native(d, M, K, N, t, w))
    nat = d.read(t["native"], M * N).reshape(M, N)
    for part in ("gather", "sums", "products"):          # breakdown first: these leave partial results
        run(part, lambda part=part: emit_strassen(d, M, K, N, t, w, parts=(part,)))
    run("strassen", lambda: emit_strassen(d, M, K, N, t, w))
    st = d.read(t["out"], M * N).reshape(M, N)

    flops = 2 * M * K * N
    say(f"{'program':34s} {'ms':>9} {'GFLOPS':>8}")
    say(f"{'native matmul + bias':34s} {timed['native']:9.2f} {flops / timed['native'] / 1e6:8.1f}")
    say(f"{'strassen: 6 strided gathers':34s} {timed['gather']:9.2f}")
    say(f"{'strassen: 9 eltwise':34s} {timed['sums']:9.2f}")
    say(f"{'strassen: 7 products (4 in place)':34s} {timed['products']:9.2f} {7 * flops / 8 / timed['products'] / 1e6:8.1f}")
    say(f"{'strassen, whole (drop-in)':34s} {timed['strassen']:9.2f} {flops / timed['strassen'] / 1e6:8.1f}   effective")
    say(f"\ndrop-in strassen / native time: {timed['strassen'] / timed['native']:.3f}   (the block-output probe measured 0.856 at this shape)")
    s_nat, s_st = snr_db(ref, nat), snr_db(ref, st)
    say(f"accuracy vs the exact product + bias: native {s_nat:.1f} dB, strassen {s_st:.1f} dB (cos {cos_sim(ref, st):.6f})")
    ok = not bool(torch.isnan(st).any()) and s_st > 30
    say("PASS" if ok else "FAIL (a low strassen SNR here points at full_matrix bias + output stride in one matmul, which no model uses yet)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
