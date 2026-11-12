"""Accuracy drop-off of recursive Strassen on the FPGA: native vs 1, 2 and 3 levels on ONE large
matmul (single engine, bf16).

    C = X . W^T      X = [M, K],  W = [N, K]      default 4096 x 7680 -> 8192

Level L replaces the product by 7^L products of size (M, K, N) / 2^L (work = (7/8)^L of native,
weight storage (7/4)^L). K and N must be multiples of 64 * 2^L, M of 2^L; K stays below 8192.

Layout trick that makes the recursion plain: X, every intermediate and C live in a recursive
block ("Z-order") layout -- a matrix is stored as its four quarters [11 | 12 | 21 | 22], each
quarter again, L deep. A quarter is then one contiguous buffer, and elementwise add / subtract
does not care about the order of elements, so every Strassen sum is one flat eltwise op and
only the 7^L leaf products see ordinary row-major blocks. The host permutes X in and C out.
The 7^L leaf weight matrices are formed on the host in fp32 and rounded to bf16 once.

Prints, per level: time, speed-up, work, and SNR / cosine vs the exact product of the bf16
operands (on a sample of rows).

4-bit weights (--weights fp4,int4,if4): every leaf weight matrix is quantized on its own, in the
repo's block format (64 values per block along K, one bf16 scale per block; fp4 = E2M1 grid,
int4 = -8..7, if4 = whichever of the two has the lower error per block). Level 0 is then the
ordinary quantized matmul, and the gap to it is what Strassen costs on top of quantization.
Accuracy is always against the exact product with the UNQUANTIZED weights.

Run from the repo root:
    python strassen/strassen_levels.py                             # the board, bf16
    python strassen/strassen_levels.py --weights fp4,int4,if4      # the board, 4-bit leaves
    python strassen/strassen_levels.py --host --weights bf16,fp4,int4,if4   # no board: numerics only
    python strassen/strassen_levels.py --selfcheck                 # host only: small-shape check
"""
import argparse
import builtins
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

BPE = 2
CHUNK = 1 << 25
ADD, SUB = "add", "sub"

_print, _quiet = builtins.print, False
builtins.print = lambda *a, **k: None if _quiet else _print(*a, **k)


def say(*a, **k):
    _print(*a, **k, flush=True)


def bf16(t):
    return t.to(torch.bfloat16).float()


def snr_db(ref, got):
    err = (ref.double() - got.double()).pow(2).sum()
    return float(10 * torch.log10(ref.double().pow(2).sum() / err.clamp_min(1e-300)))


def cos_sim(ref, got):
    a, b = ref.double().flatten(), got.double().flatten()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


# ----------------------------------------------------------------------------- 4-bit weights

FP4_VALUES = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_CODES = [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7]


def quant4(W, kind, dev):
    """quant_lib.quantize_if4 on `dev` (same bytes; --selfcheck compares them).
    W [N, K] -> (packed codes uint8 [N, K/2], scales bf16 [N, K/64], dequantized fp32 [N, K] on dev).
    Scale sign selects the grid on the chip: negative = int4, positive = fp4."""
    N, K = W.shape
    vals = torch.tensor(FP4_VALUES, device=dev)
    codes = torch.tensor(FP4_CODES, device=dev, dtype=torch.uint8)
    packed, scales, deq = [], [], []
    step = max(1, (1 << 21) // K)
    for r in range(0, N, step):
        b = bf16(W[r: r + step].to(dev)).view(-1, K // 64, 64)
        amax = b.abs().amax(-1).clamp(min=1e-8)
        si = bf16(amax / 7.0)[..., None]
        qi = (b / si).round().clamp(-8, 7)
        sf = bf16(amax / 6.0)[..., None]
        idx = ((b / sf)[..., None] - vals).abs().argmin(-1)
        ri, rf = qi * si, vals[idx] * sf
        if kind == "if4":
            use_i = (b - ri).pow(2).sum(-1) <= (b - rf).pow(2).sum(-1)
        else:
            use_i = torch.full(amax.shape, kind == "int4", device=dev)
        u = use_i[..., None]
        nib = torch.where(u, (qi.to(torch.int16) & 0xF).to(torch.uint8), codes[idx]).view(-1, K)
        packed.append((nib[:, 0::2] | (nib[:, 1::2] << 4)).cpu())
        scales.append(torch.where(use_i, -si[..., 0], sf[..., 0]).to(torch.bfloat16).cpu())
        deq.append(torch.where(u, ri, rf).view(-1, K))
    return torch.cat(packed), torch.cat(scales), torch.cat(deq)


# ----------------------------------------------------------------------------- host side

def z_pack(t, L):
    """[rows, cols] -> flat recursive block layout, L deep (quarters in order 11, 12, 21, 22)."""
    if L == 0:
        return t.reshape(-1)
    r, c = t.shape[0] // 2, t.shape[1] // 2
    return torch.cat([z_pack(q, L - 1) for q in (t[:r, :c], t[:r, c:], t[r:, :c], t[r:, c:])])


def z_unpack(flat, rows, cols, L):
    if L == 0:
        return flat.reshape(rows, cols)
    r, c, q = rows // 2, cols // 2, flat.numel() // 4
    b = [z_unpack(flat[i * q: (i + 1) * q], r, c, L - 1) for i in range(4)]
    return torch.cat([torch.cat([b[0], b[1]], 1), torch.cat([b[2], b[3]], 1)], 0)


def weight_leaves(W, L):
    """The 7^L leaf weight matrices [N / 2^L, K / 2^L], in the order the emitter visits them.
    Yij = block (i, j) of W^T, i.e. W[j-th N half, i-th K half]."""
    if L == 0:
        return [W]
    n, k = W.shape[0] // 2, W.shape[1] // 2
    Y11, Y12, Y21, Y22 = W[:n, :k], W[n:, :k], W[:n, k:], W[n:, k:]
    out = []
    for combo in (Y11 + Y22, Y11, Y12 - Y22, Y21 - Y11, Y22, Y11 + Y12, Y21 + Y22):
        out += weight_leaves(combo, L - 1)
    return out


def emit(be, L, depth, x, leaf0, out, m, k, n, pools, wbase):
    """Recursive Strassen on backend `be` (device or host model). x: [m, k] in block layout L deep;
    out: [m, n] in block layout L deep. pools[depth] = (5 x-sum buffers, 7 product buffers, temp)."""
    if L == 0:
        be.mm(m, k, n, x, wbase + leaf0 * n * k * BPE, out)
        return
    h, kk, nn = m // 2, k // 2, n // 2
    qx, qo = h * kk * BPE, h * nn * BPE
    x11, x12, x21, x22 = x, x + qx, x + 2 * qx, x + 3 * qx
    S, P, T = pools[depth]
    for (a, b, mode), o in zip([(x11, x22, ADD), (x21, x22, ADD), (x11, x12, ADD), (x21, x11, SUB), (x12, x22, SUB)], S):
        be.elt(h, kk, a, b, o, mode)
    step = 7 ** (L - 1)
    for i, src in enumerate([S[0], S[1], x11, x22, S[2], S[3], S[4]]):
        emit(be, L - 1, depth + 1, src, leaf0 + i * step, P[i], h, kk, nn, pools, wbase)
    c11, c12, c21, c22 = out, out + qo, out + 2 * qo, out + 3 * qo
    e = lambda a, b, o, mode: be.elt(h, nn, a, b, o, mode)
    e(P[0], P[3], T, ADD), e(T, P[4], T, SUB), e(T, P[6], c11, ADD)
    e(P[2], P[4], c12, ADD)
    e(P[1], P[3], c21, ADD)
    e(P[0], P[1], T, SUB), e(T, P[2], T, ADD), e(T, P[5], c22, ADD)


def pool_plan(M, K, N, levels):
    """Element counts of the scratch at each depth: [(x-sum size, product size)] for depth 0..levels-1."""
    return [((M >> (d + 1)) * (K >> (d + 1)), (M >> (d + 1)) * (N >> (d + 1))) for d in range(levels)]


class HostModel:
    """The device ops on a flat fp32 array, rounding to bf16 on every write (accumulation in fp32)."""

    def __init__(self, elements, device):
        self.mem = torch.zeros(elements, device=device)

    def wr(self, addr, t):
        t = bf16(t.reshape(-1))
        self.mem[addr // BPE: addr // BPE + t.numel()] = t.to(self.mem.device)

    def rd(self, addr, n):
        return self.mem[addr // BPE: addr // BPE + n]

    def mm(self, m, k, n, a, w, out):
        self.wr(out, self.rd(a, m * k).reshape(m, k) @ self.rd(w, n * k).reshape(n, k).T)

    def elt(self, rows, cols, a, b, out, mode):
        x, y = self.rd(a, rows * cols), self.rd(b, rows * cols)
        self.wr(out, x + y if mode == ADD else x - y)


class Layout:
    """Bump allocation of every buffer one run needs, from two bases (weights, tensors)."""

    def __init__(self, M, K, N, max_level, wbase, tbase):
        self.w = wbase
        leaves = max((7 ** L) * (N >> L) * (K >> L) for L in range(max_level + 1))
        self.w_bytes = leaves * BPE
        t = tbase
        self.x, t = t, t + M * K * BPE
        self.out, t = t, t + M * N * BPE
        self.pools = []
        for sx, sp in pool_plan(M, K, N, max_level):
            S = [t + i * sx * BPE for i in range(5)]
            t += 5 * sx * BPE
            P = [t + i * sp * BPE for i in range(7)]
            t += 7 * sp * BPE
            self.pools.append((S, P, t))
            t += sp * BPE
        self.t_bytes = t - tbase


def host_run(X, W, L, wt, lay, dev):
    """The whole level-L product on the host model (lay built with bases 0, 0). Leaves are bf16 or
    dequantized 4-bit (code x bf16 scale, not re-rounded); products accumulate in fp32, bf16 out."""
    M, K, N = X.shape[0], W.shape[1], W.shape[0]
    leaves = weight_leaves(W, L)
    n, k = leaves[0].shape
    wm = torch.empty(len(leaves), n, k, device=dev)     # weights and tensors are separate address spaces here
    for i, leaf in enumerate(leaves):
        wm[i] = bf16(leaf.to(dev)) if wt == "bf16" else quant4(leaf, wt, dev)[2]
    del leaves
    tm = HostModel(lay.t_bytes // BPE, dev)

    class Both:
        mm = staticmethod(lambda m, kk, nn, a, w, out: tm.wr(out, tm.rd(a, m * kk).reshape(m, kk) @ wm[w // (nn * kk * BPE)].T))
        elt = staticmethod(tm.elt)

    tm.wr(lay.x, z_pack(X, L))
    emit(Both, L, 0, lay.x, 0, lay.out, M, K, N, lay.pools, 0)
    got = z_unpack(tm.rd(lay.out, M * N).cpu(), M, N, L)
    del wm, tm
    if dev == "cuda":
        torch.cuda.empty_cache()
    return got


def selfcheck(args):
    """Host only: the recursion, layouts and leaf order reproduce the plain product, and quant4
    writes the same bytes as quant_lib."""
    import quant_lib
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M, K, N = 64, 1024, 512
    g = torch.Generator().manual_seed(args.seed)
    X, W = bf16(torch.randn(M, K, generator=g)), bf16(torch.randn(N, K, generator=g) / K ** 0.5)
    ref = X.double() @ W.double().T
    lay = Layout(M, K, N, 3, 0, 0)
    ok = True
    for wt, variant in (("fp4", False), ("int4", True), ("if4", None)):
        codes, scales, deq = quant4(W, wt, dev)
        data, sc = quant_lib.quantize_if4(W.to(torch.bfloat16), block_size=64, int_variant=variant)
        same = codes.numpy().tobytes() == data and scales.view(torch.int16).numpy().tobytes() == sc
        lib = quant_lib.dequantize_if4(data, sc, N, K).float()
        ok &= same
        say(f"{wt:>4}: bytes {'match' if same else 'DIFFER from'} quant_lib, weight snr {snr_db(W, deq.cpu()):5.1f} dB "
            f"(quant_lib dequant vs ours {snr_db(lib, deq.cpu()):5.1f} dB)")
    for wt, floor in (("bf16", 25), ("fp4", 0), ("int4", 0), ("if4", 0)):
        for L in range(4):
            got = host_run(X, W, L, wt, lay, dev)
            s = snr_db(ref, got)
            ok &= s > floor
            say(f"{wt:>4} level {L}: {7 ** L:4d} products, host model snr {s:5.1f} dB, cos {cos_sim(ref, got):.6f}")
    say("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ----------------------------------------------------------------------------- device

class Device:
    def __init__(self, dev):
        global _quiet
        import user_dma_core as U
        self.U = U
        _quiet = True
        try:
            ue = self.ue = U.UnifiedEngine(**({"device_name": dev} if dev else {}))   # default DRAM map
            halt = bytes(64)
            ue.dma_write(ue.h2c_device, U.DRAM_INSTRUCTION_ADDR, halt, len(halt))
        finally:
            _quiet = False
        self.wbase, self.tbase, self.prog = U.DRAM_START_ADDR, U.DRAM_ACTIVATION_ADDR, ue.get_program_dram_addr()
        self.w_room, self.t_room = self.tbase - self.wbase, U.DRAM_INSTRUCTION_ADDR - self.tbase
        period = getattr(ue, "_clock_period_ns", 0) or 0
        self.peak = 128 * 1e3 / period / 1e3 if period else 0.0
        self.regs = {}
        self.quant = False

    def raw(self, addr, t):
        """Bytes as they are (4-bit codes, block scales)."""
        b = t.contiguous().view(-1).view(torch.uint8)
        for off in range(0, b.numel(), CHUNK):
            part = b[off: off + CHUNK]
            self.ue.dma_write(self.ue.h2c_device, addr + off, part, part.numel())

    def load_weights(self, leaves, wt, cdev):
        """bf16: the leaves back to back. 4-bit: all code planes, then all scale planes."""
        self.quant = wt != "bf16"
        if not self.quant:
            self.upload(self.wbase, torch.stack(leaves))
            return
        n, k = leaves[0].shape
        self.sbase = self.wbase + -(-len(leaves) * n * k // 2 // 128) * 128
        for i, leaf in enumerate(leaves):
            codes, scales, _ = quant4(leaf, wt, cdev)
            self.raw(self.wbase + i * n * k // 2, codes)
            self.raw(self.sbase + i * n * k // 32, scales)

    def upload(self, addr, t):
        flat = t.to(torch.bfloat16).contiguous().flatten()
        for off in range(0, flat.numel(), CHUNK):
            self.ue.dma_to_accelerator_memory(addr + off * BPE, flat[off: off + CHUNK])

    def zero(self, addr, elements):
        for off in range(0, elements, CHUNK):
            self.ue.dma_to_accelerator_memory(addr + off * BPE, torch.zeros(min(CHUNK, elements - off), dtype=torch.bfloat16))

    def read(self, addr, elements):
        out = [self.ue.dma_from_accelerator_memory(addr + off * BPE, (min(CHUNK, elements - off),)).float()
               for off in range(0, elements, CHUNK)]
        return torch.cat(out)

    def _reg(self, rows):
        return self.regs[rows]

    def mm(self, m, k, n, a, w, out):
        if self.quant:      # w is the leaf's address in the bf16 layout; map it to its code / scale planes
            i = (w - self.wbase) // (n * k * BPE)
            self.ue.quantized_matmat_core(M=m, K=k, N=n, A_DRAM_ADDR=a, B_DRAM_ADDR=self.wbase + i * n * k // 2,
                                          OUTPUT_DRAM_ADDR=out, SCALE_DRAM_ADDR=self.sbase + i * n * k // 32,
                                          data_type=self.U.TYPE.IF4, gpr_M_reg=self._reg(m))
            return
        self.ue.matmat_mul_core(M=m, K=k, N=n, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out,
                                C_DRAM_ADDR=None, bias_mode="broadcast_N", gpr_M_reg=self._reg(m))

    def elt(self, rows, cols, a, b, out, mode):
        M = self.U.UE_MODE
        self.ue.eltwise_core_dram(M=rows, N=cols, dram_a=a, dram_b=b, dram_out=out,
                                  mode=M.ELTWISE_ADD if mode == ADD else M.ELTWISE_SUB, gpr_M_reg=self._reg(rows))

    def compile(self, row_counts, body):
        """One resident program at the engine's program pointer (loops resolve against it)."""
        global _quiet
        ue = self.ue
        _quiet = True
        try:
            ue.clear_inst_id()
            ue.start_capture()
            ue.reset_isa_reg_counter()
            self.regs = {}
            for rows in sorted(set(row_counts)):        # one GPR per distinct row count in the program
                self.regs[rows] = ue.alloc_isa_reg()
                ue.generate_instruction_add_set(self.regs[rows], rows)
            body()
            ue.stop_capture()
            ue.generate_instruction_halt()
            size = ue.get_capture_instruction_size_bytes()
            ue.write_captured_instructions_to_dram(self.prog)
            ue.clear_capture_buffer()
        finally:
            _quiet = False
        return size

    def run(self, timeout):
        global _quiet
        ue = self.ue
        _quiet = True
        try:
            t0 = time.perf_counter()
            ue.start_execute_from_dram(self.prog)
            ue.wait_queue(timeout)
            wall = time.perf_counter() - t0
            us = ue.report_latency_in_us()
        finally:
            _quiet = False
        period = getattr(ue, "_clock_period_ns", 0) or 0
        if period:
            wrap = (2 ** 32) * period / 1e3
            while us + wrap / 2 < wall * 1e6:
                us += wrap
        return us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--K", type=int, default=7680, help="largest K below 8192 that splits three times (15 x 512)")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--sample-rows", type=int, default=256, help="rows of C compared against the exact product")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="bf16", help="comma list of bf16, fp4, int4, if4 (if4 = per-block best of fp4 / int4)")
    ap.add_argument("--host", action="store_true", help="no board: accuracy from the host model at the full shape")
    ap.add_argument("--selfcheck", action="store_true", help="host only: validate the recursion on a small shape")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args)
    M, K, N = args.M, args.K, args.N
    levels = sorted({int(v) for v in args.levels.split(",")})
    wts = args.weights.split(",")
    assert all(w in ("bf16", "fp4", "int4", "if4") for w in wts), f"--weights: {wts}"
    top = max(levels)
    assert M % (1 << top) == 0 and K % (64 << top) == 0 and N % (64 << top) == 0, \
        f"{top} levels need M a multiple of {1 << top} and K, N multiples of {64 << top}"
    assert K < 8192, "K >= 8192 takes matmat_mul_core's 16-column fallback"

    g = torch.Generator().manual_seed(args.seed)
    X = bf16(torch.randn(M, K, generator=g))
    W = bf16(torch.randn(N, K, generator=g) / K ** 0.5)
    rows = torch.randperm(M, generator=g)[: min(args.sample_rows, M)].sort().values
    cdev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = (X[rows].to(cdev).double() @ W.to(cdev).double().T).cpu()

    note = f"accuracy is measured on {len(rows)} rows against the exact product of the bf16 operands (unquantized weights)\n"
    if args.host:
        lay = Layout(M, K, N, top, 0, 0)
        say(f"shape [{M}, {K}] x [{K} -> {N}], host model on {cdev}, no board")
        say(note)
        say(f"{'weights':>7} {'level':>5} {'products':>8} {'leaf shape':>18} {'work':>6} {'snr dB':>7} {'vs level 0':>10} {'cos':>9}")
        for wt in wts:
            base = None
            for L in levels:
                got = host_run(X, W, L, wt, lay, cdev)[rows]
                s = snr_db(ref, got)
                base = s if base is None and L == 0 else base
                drop = f"{s - base:+9.1f}" if base is not None else "      n/a"
                say(f"{wt:>7} {L:5d} {7 ** L:8d} {f'{M >> L} x {K >> L} x {N >> L}':>18} {100 * (7 / 8) ** L:5.1f}% "
                    f"{s:7.1f} {drop:>10} {cos_sim(ref, got):9.6f}")
        return

    d = Device(args.dev)
    lay = Layout(M, K, N, top, d.wbase, d.tbase)
    assert lay.w_bytes <= d.w_room, f"weights need {lay.w_bytes >> 20} MB, region is {d.w_room >> 20} MB"
    assert lay.t_bytes <= d.t_room, f"tensors need {lay.t_bytes >> 20} MB, region is {d.t_room >> 20} MB"
    d.zero(d.tbase, lay.t_bytes // BPE)                 # fresh DRAM holds NaN patterns
    say(f"shape [{M}, {K}] x [{K} -> {N}], one engine, peak {d.peak:.1f} GFLOPS; "
        f"weights up to {lay.w_bytes >> 20} MB (bf16), tensors {lay.t_bytes >> 20} MB")
    say(note)
    say(f"{'weights':>7} {'level':>5} {'products':>8} {'leaf shape':>18} {'work':>6} {'time ms':>10} {'vs native':>9} {'snr dB':>7} {'cos':>9}")
    for wt in wts:
        base_ms = None
        for L in levels:
            leaves = weight_leaves(W, L)
            d.load_weights(leaves, wt, cdev)
            del leaves
            d.upload(lay.x, z_pack(X, L))
            row_counts = [M >> j for j in range(1, L + 1)] or [M]
            d.compile(row_counts, lambda: emit(d, L, 0, lay.x, 0, lay.out, M, K, N, lay.pools, d.wbase))
            say(f"{wt:>7} {L:5d} {7 ** L:8d}  running...", end="\r")
            try:
                us = d.run(60.0 + 2 * M * K * N / 2e9)
            except Exception as exc:
                say(f"{wt:>7} {L:5d} {7 ** L:8d}  FAILED: {type(exc).__name__}: {str(exc)[:110]}")
                say("stopping: a timed-out program leaves the engine stuck; reset the board before the next run")
                sys.exit(1)
            got = z_unpack(d.read(lay.out, M * N), M, N, L)[rows]
            ms = us / 1e3
            base_ms = ms if base_ms is None and L == 0 else base_ms
            rel = f"{ms / base_ms:8.3f}x" if base_ms else "      n/a"
            say(f"{wt:>7} {L:5d} {7 ** L:8d} {f'{M >> L} x {K >> L} x {N >> L}':>18} {100 * (7 / 8) ** L:5.1f}% "
                f"{ms:10.1f} {rel} {snr_db(ref, got):7.1f} {cos_sim(ref, got):9.6f}")
    say("\nwork = matmul FLOPs relative to native; 'vs native' is against level 0 of the same weight type.")


if __name__ == "__main__":
    main()
