"""Native matmul vs one level of Strassen on the FPGA, swept over (M, K, N). Single engine, bf16.

    C = X . W^T      X = [M, K] activations,  W = [N, K] weights  ->  C = [M, N]

Native:   one matmat_mul_core.
Strassen: halves (h, k, n) = (M/2, K/2, N/2); the seven weight combinations are formed on the
          host (1.75x the weight storage); on the device 5 activation sums, 7 products, 8 output
          sums. X is supplied as its two column halves and C is left as four blocks -- stitching
          those back into [M, N] rows is NOT timed.

Every shape is timed both ways (hardware cycle counter) and both results are checked against the
exact product of the bf16 operands on a sample of rows. Results append to a CSV as they arrive
(a crashed run keeps what it measured; --resume continues it) and a bar-chart PNG is drawn at the
end (or any time with --plot-only).

Shapes: M even, K and N multiples of 64. A half that is not a 64-multiple (K or N = 64, 192, ...)
is zero-padded up to one, because an SRAM row is 64 elements -- that padding is extra work, and
at K = 64 or N = 64 it makes Strassen do MORE matmul work than native. Default limits follow
what user_hw_test.py exercises: K <= 8064 (8192 and up forces matmat_mul_core's 16-column strip
fallback), M <= 4096, N <= 8192. The full lattice is millions of shapes, so the sweep is a grid:
    --grid pow2    10 x 8 x 8 = 640 shapes   (default, roughly 15 minutes)
    --grid dense   adds the 1.5x midpoints   17 x 14 x 14 = 3332 shapes
A hung program leaves the engine stuck, so the sweep stops at the first timeout: reset the board
and rerun with --resume (it skips every shape already in the CSV, the failed one included).
or give your own lists with --m / --k / --n.

Run from the repo root:
    python strassen/strassen_sweep.py
    python strassen/strassen_sweep.py --grid dense --resume
    python strassen/strassen_sweep.py --plot-only
"""
import argparse
import builtins
import csv
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

BPE = 2
PARAMS_BASE, TENSOR_BASE, PROGRAM_BASE = 0x80000000, 0x98000000, 0xD0000000
PROGRAM_SLOT = 256 << 20           # largest program the sweep may capture
CHUNK = 1 << 25                    # elements per DMA
FIELDS = ["M", "K", "N", "native_ms", "strassen_ms", "ratio", "native_gflops", "strassen_gflops",
          "peak_gflops", "native_snr_db", "strassen_snr_db", "native_gflop", "strassen_gflop", "flop_cut_pct",
          "strassen_real_gflops", "status"]
GRIDS = {
    # K and N go down to 64. A half below 64 lanes cannot exist on the device (one SRAM row is 64
    # elements), so every Strassen half is padded up to a 64-multiple: at K = 64 the seven products
    # each run at the full 64 width, i.e. MORE work than native. The FLOP columns show that.
    "pow2": dict(M=[2, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                 K=[64, 128, 256, 512, 1024, 2048, 4096, 8064],
                 N=[64, 128, 256, 512, 1024, 2048, 4096, 8192]),
    "dense": dict(M=[2, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
                  K=[64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8064],
                  N=[64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]),
}


def pad64(v):
    return -(-v // 64) * 64


LEVEL = 1                          # Strassen recursion depth (--strassen-level)
DEEP_PARAMS_BASE = 0x40000000      # levels 2-3 keep up to 5.4x the weights: start the weight region lower


def padl(v):
    """K or N padded so it splits LEVEL times into 64-lane-aligned pieces."""
    q = 64 << LEVEL
    return -(-v // q) * q


def work(M, K, N):
    """(native FLOPs, Strassen FLOPs): matmul work actually executed, padding included."""
    return 2 * M * K * N, 7 ** LEVEL * 2 * (M >> LEVEL) * (padl(K) >> LEVEL) * (padl(N) >> LEVEL)


def complete(r):
    """Fill the derived columns of a CSV row (also upgrades rows written before they existed)."""
    if r.get("status") != "ok":
        return r
    nat, st = work(int(r["M"]), int(r["K"]), int(r["N"]))
    ms = float(r["strassen_ms"])
    r.update(native_gflop=round(nat / 1e9, 4), strassen_gflop=round(st / 1e9, 4),
             flop_cut_pct=round(100 * (1 - st / nat), 2), strassen_real_gflops=round(st / ms / 1e6, 4) if ms else 0.0)
    return r


_print, _quiet = builtins.print, False
builtins.print = lambda *a, **k: None if _quiet else _print(*a, **k)


def say(*a, **k):
    _print(*a, **k, flush=True)


def bf16(t):
    return t.to(torch.bfloat16).float()


def snr_db(ref, got):
    err = (ref.double() - got.double()).pow(2).sum()
    return float(10 * torch.log10(ref.double().pow(2).sum() / err.clamp_min(1e-300)))


# ----------------------------------------------------------------------------- device

class Bench:
    """One engine, buffers sized once for the largest shape and reused by every shape."""

    def __init__(self, dev, mM, mK, mN, engines=1):
        global _quiet
        import user_dma_core as U
        self.U, self.NE, self.sched = U, engines, None
        pbase = PARAMS_BASE if LEVEL == 1 else DEEP_PARAMS_BASE
        bases = dict(params_dram_base=pbase, tensor_dram_base=TENSOR_BASE, program_dram_base=PROGRAM_BASE)
        windows = None
        if engines > 1:     # N private 512 MiB windows from 0, then this probe's map (same offsets) above them
            from multi_engine_shard import MultiEngineScheduler, PrivateArena, model_multicore_layout
            windows, model_base = model_multicore_layout(engines, model_bytes=4 << 30)
            bases = {key: v - pbase + model_base for key, v in bases.items()}
        kw = dict(bases)
        if dev:
            kw["device_name"] = dev
        _quiet = True
        try:
            ue = self.ue = U.UnifiedEngine(**kw)        # before any upload: its DRAM self-test stomps DRAM start
            halt = bytes(64)
            ue.dma_write(ue.h2c_device, U.DRAM_INSTRUCTION_ADDR, halt, len(halt))
            if engines > 1:
                # scheduler + flag clear before any upload too: both can reset an engine
                arena = PrivateArena(engines, windows=windows, isa_bytes=256 << 20, tensor_bytes=128 << 20)
                self.sched = MultiEngineScheduler(ue, num_engines=engines, arena=arena, allow_unaligned_rows=True,
                                                  handshake="four_phase", region_rendezvous="master_worker",
                                                  allow_more_than_two_engines=True)
                self.sched.preclear_flags()
                for w in self.sched.workers:
                    w.allocate_params_dram(64 << 10)
        finally:
            _quiet = False
        self.engs = self.sched.engines if self.sched else [ue]
        self.program_limit = bases["params_dram_base"] + (4 << 30)
        h, k, n = mM // 2, pad64(mK // 2), pad64(mN // 2)
        if LEVEL == 1:
            need_p, need_t = (mN * mK + 7 * n * k) * BPE, (mM * mK * 2 + mM * mN + 5 * h * k + 12 * h * n) * BPE
        else:               # recursive path: leaf weights + block-layout X / C / scratch (strassen_levels.Layout)
            from strassen_levels import Layout
            unit = 1 << LEVEL                            # every engine gets its own block-layout X / C / scratch
            self.rows_cap = -(-mM // (engines * unit)) * unit
            probe = Layout(self.rows_cap, padl(mK), padl(mN), LEVEL, 0, 0)
            leaves = Layout(mM, padl(mK), padl(mN), LEVEL, 0, 0).w_bytes
            need_p, need_t = mN * mK * BPE + leaves, (mM * mK + mM * mN) * BPE + engines * probe.t_bytes
        assert need_p <= TENSOR_BASE - pbase, f"weights need {need_p >> 20} MB, region is {(TENSOR_BASE - pbase) >> 20} MB"
        assert need_t <= PROGRAM_BASE - TENSOR_BASE, f"tensors need {need_t >> 20} MB, region is {(PROGRAM_BASE - TENSOR_BASE) >> 20} MB"
        self.w_full = self._params(mN * mK)
        self.x_full = self._tensor(mM * mK)
        self.out = self._tensor(mM * mN)
        if LEVEL == 1:
            self.w_c = [self._params(n * k) for _ in range(7)]
            self.xl, self.xr = self._tensor(mM * k), self._tensor(mM * k)
            self.s = [self._tensor(h * k) for _ in range(5)]
            self.p = [self._tensor(h * n) for _ in range(7)]
            self.c = [self._tensor(h * n) for _ in range(4)]
            self.t = self._tensor(h * n)
        else:
            wl = self._params(leaves // BPE)             # leaf weights: shared, read-only
            self.lays = [Layout(self.rows_cap, padl(mK), padl(mN), LEVEL, wl, self._tensor(probe.t_bytes // BPE))
                         for _ in range(engines)]
            self.lay = self.lays[0]
        self.prog = ue.get_program_dram_addr()
        self.prog0 = [(e, e.get_program_dram_addr()) for e in self.engs]   # compile() rewinds to here
        period = getattr(ue, "_clock_period_ns", 0) or 0
        self.peak = engines * 128 * 1e3 / period / 1e3 if period else 0.0      # GFLOPS, all engines
        say(f"{engines} engine{'s' if engines > 1 else ''} up: weights {need_p >> 20} MB, tensors {need_t >> 20} MB, peak {self.peak:.1f} GFLOPS")

    def _params(self, elements):
        addr = self.ue.get_params_dram_addr()
        self.ue.allocate_params_dram(elements * BPE)
        return addr

    def _tensor(self, elements):       # zero-filled: fresh DRAM holds NaN patterns and ops may over-read
        addr = self.ue.allocate_tensor_dram(elements * BPE)
        for off in range(0, elements, CHUNK):
            self.ue.dma_to_accelerator_memory(addr + off * BPE, torch.zeros(min(CHUNK, elements - off), dtype=torch.bfloat16))
        return addr

    def upload(self, addr, t):
        flat = t.to(torch.bfloat16).contiguous().flatten()
        for off in range(0, flat.numel(), CHUNK):
            self.ue.dma_to_accelerator_memory(addr + off * BPE, flat[off: off + CHUNK])

    def read(self, addr, shape):
        return self.ue.dma_from_accelerator_memory(addr, shape).float()

    def split(self, total):
        """[(offset, count)] per engine: rows as even as they go (an engine may get 0)."""
        base, rem = divmod(total, self.NE)
        counts = [base + (i < rem) for i in range(self.NE)]
        return [(sum(counts[:i]), counts[i]) for i in range(self.NE)]

    def compile(self, total, emit, shards=None):
        """Capture one program: every engine runs emit(ue, r_rows, row0, rows) on its own slice of
        `total` rows, in a single fenced round (one engine: the whole range, no fences).

        One engine: the program is rewritten at the engine's program pointer every time. Loop
        instructions are resolved against that pointer at capture time, so a program written
        anywhere else jumps into the wrong code and hangs the engine.
        Several engines: programs are allocated one after another (the scheduler places the
        workers' the same way), as the model tests do."""
        global _quiet
        ue, S = self.ue, self.sched
        shards = shards or self.split(total)

        def body(e):
            row0, rows = shards[e]
            if rows == 0:
                return
            eng = self.engs[e]
            eng.reset_isa_reg_counter()
            r = eng.alloc_isa_reg()
            eng.generate_instruction_add_set(r, rows)
            emit(eng, r, row0, rows)

        for eng, addr in self.prog0:                    # one resident program at a time: reuse the space,
            eng._next_program_dram_addr = addr          # or a long sweep fills each engine's 256 MB slice
        _quiet = True
        try:
            if S is None:
                ue.clear_inst_id()
            ue.start_capture()
            if S is None:
                body(0)
            else:
                S.begin_program()
                S.release()
                body(0)
                for e in S.worker_indices():
                    S.begin_worker_round(e)
                    body(e)
                    S.end_worker_round(e)
                S.join()
                self.wprog = S.finalize()
            ue.stop_capture()
            ue.generate_instruction_halt()
            size = ue.get_capture_instruction_size_bytes()
            assert size <= PROGRAM_SLOT, f"program is {size >> 20} MB, limit is {PROGRAM_SLOT >> 20} MB"
            addr = ue.get_program_dram_addr()
            assert addr + size <= self.program_limit, "program arena is full -- sweep fewer shapes per run (--resume continues)"
            ue.write_captured_instructions_to_dram(addr)
            if S is not None:
                ue.allocate_program_dram(size)
            ue.clear_capture_buffer()
            self.prog = addr
        finally:
            _quiet = False

    def run(self, timeout=900.0):
        """Execute the resident program; returns microseconds from the hardware cycle counter."""
        global _quiet
        ue = self.ue
        _quiet = True
        try:
            if self.sched is not None:                  # workers first, every run: they halt with the program
                self.sched.start_workers(self.wprog)
            t0 = time.perf_counter()
            ue.start_execute_from_dram(self.prog)
            ue.wait_queue(timeout)
            wall = time.perf_counter() - t0
            us = ue.report_latency_in_us()
        finally:
            _quiet = False
        period = getattr(ue, "_clock_period_ns", 0) or 0
        if period:                                      # the 32-bit cycle counter wraps
            wrap = (2 ** 32) * period / 1e3
            while us + wrap / 2 < wall * 1e6:
                us += wrap
        return us

    @staticmethod
    def mm(ue, r, M, K, N, a, w, out):
        ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a, B_DRAM_ADDR=w, OUTPUT_DRAM_ADDR=out,
                           C_DRAM_ADDR=None, bias_mode="broadcast_N", gpr_M_reg=r)

    @staticmethod
    def elt(ue, r, rows, cols, a, b, out, mode):
        ue.eltwise_core_dram(M=rows, N=cols, dram_a=a, dram_b=b, dram_out=out, mode=mode, gpr_M_reg=r)


def measure_deep(b, X, W, M, K, N, runs):
    """measure() for LEVEL >= 2: the recursive emitter of strassen_levels.py (block layout).
    Several engines: the rows are cut into one contiguous range per engine (a multiple of 2^LEVEL
    rows each) and every engine runs a complete, independent recursion on its own sub-matrix, in
    its own buffers -- one fenced round, shared read-only leaf weights."""
    from strassen_levels import emit, z_pack, z_unpack
    L, h = LEVEL, M // 2
    unit = 1 << L
    base, rem = divmod(M // unit, b.NE)
    counts = [(base + (i < rem)) * unit for i in range(b.NE)]
    shards = [(sum(counts[:i]), counts[i]) for i in range(b.NE)]
    owner = {r0: e for e, (r0, c) in enumerate(shards) if c}
    Kp, Np = padl(K), padl(N)
    MODE = {"add": b.U.UE_MODE.ELTWISE_ADD, "sub": b.U.UE_MODE.ELTWISE_SUB}

    def native(ue, r, r0, rows):
        b.mm(ue, r, rows, K, N, b.x_full + r0 * K * BPE, b.w_full, b.out + r0 * N * BPE)

    def strassen(ue, r, r0, rows):
        lay, regs = b.lays[owner[r0]], {}
        for j in range(1, L + 1):                       # one GPR per row count the recursion uses
            regs[rows >> j] = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(regs[rows >> j], rows >> j)

        class BE:
            mm = staticmethod(lambda m, k, n, a, w, out: b.mm(ue, regs[m], m, k, n, a, w, out))
            elt = staticmethod(lambda rows_, cols, a, d, out, mode: b.elt(ue, regs[rows_], rows_, cols, a, d, out, MODE[mode]))

        emit(BE, L, 0, lay.x, 0, lay.out, rows, Kp, Np, lay.pools, lay.w)

    limit = 20.0 + 2 * M * K * N / 2e9
    rows = min(h, 32)
    idx = torch.cat([torch.arange(rows), torch.arange(h, h + rows)])
    ref = X[idx, :K].double() @ W.double().T
    for e, (r0, c) in enumerate(shards):
        if c:
            b.upload(b.lays[e].x, z_pack(torch.nn.functional.pad(X[r0: r0 + c, :K], (0, Kp - K)), L))
    say("native.. ", end="")
    b.compile(M, native)
    us_n = min(b.run(limit) for _ in range(runs))
    got_n = torch.cat([b.read(b.out, (rows, N)), b.read(b.out + h * N * BPE, (rows, N))])
    say("\b\b\b\b\b\b\b\b\bstrassen.", end="")
    b.compile(M, strassen, shards=shards)
    us_s = min(b.run(limit) for _ in range(runs))
    say("\b\b\b\b\b\b\b\b\b", end="")
    full = torch.cat([z_unpack(b.read(b.lays[e].out, (c * Np,)), c, Np, L) for e, (r0, c) in enumerate(shards) if c])
    got_s = full[idx][:, :N]
    flops = 2 * M * K * N
    return dict(native_ms=us_n / 1e3, strassen_ms=us_s / 1e3, ratio=us_s / us_n if us_n else float("nan"),
                native_gflops=flops / us_n / 1e3 if us_n else 0.0, strassen_gflops=flops / us_s / 1e3 if us_s else 0.0,
                native_snr_db=snr_db(ref, got_n), strassen_snr_db=snr_db(ref, got_s))


def measure(b, X, W, M, K, N, runs):
    """Time native and Strassen for one shape (X and W for this K, N are already on the device)."""
    if LEVEL > 1:
        return measure_deep(b, X, W, M, K, N, runs)
    h, n_real = M // 2, N // 2
    k, n = pad64(K // 2), pad64(N // 2)                 # device widths of the halves (zero-padded)
    ADD, SUB = b.U.UE_MODE.ELTWISE_ADD, b.U.UE_MODE.ELTWISE_SUB
    s, p, c, t = b.s, b.p, b.c, b.t

    # Sharding (several engines): both versions split ROWS, one fenced round, no barrier inside.
    #   native    rows [r0, r0 + rows) of X and of C.
    #   Strassen  half-row indices [r0, r0 + rows): row i of the top half and row i of the bottom
    #             half meet in every x-sum, product and c-sum, and nothing else does -- so one
    #             engine runs the whole 5 + 7 + 8 pipeline on its slice and never waits.
    def native(ue, r, r0, rows):
        b.mm(ue, r, rows, K, N, b.x_full + r0 * K * BPE, b.w_full, b.out + r0 * N * BPE)

    def strassen(ue, r, r0, rows):
        xo, no = r0 * k * BPE, r0 * n * BPE
        x11, x21 = b.xl + xo, b.xl + h * k * BPE + xo   # the halves are stored at row pitch k
        x12, x22 = b.xr + xo, b.xr + h * k * BPE + xo
        S, P, C, T = [a + xo for a in s], [a + no for a in p], [a + no for a in c], t + no
        for (a, d, mode), out in zip([(x11, x22, ADD), (x21, x22, ADD), (x11, x12, ADD), (x21, x11, SUB), (x12, x22, SUB)], S):
            b.elt(ue, r, rows, k, a, d, out, mode)
        for a, w, out in zip([S[0], S[1], x11, x22, S[2], S[3], S[4]], b.w_c, P):
            b.mm(ue, r, rows, k, n, a, w, out)
        e = lambda a, d, out, mode: b.elt(ue, r, rows, n, a, d, out, mode)
        e(P[0], P[3], T, ADD), e(T, P[4], T, SUB), e(T, P[6], C[0], ADD)     # C11
        e(P[2], P[4], C[1], ADD)                                             # C12
        e(P[1], P[3], C[2], ADD)                                             # C21
        e(P[0], P[1], T, SUB), e(T, P[2], T, ADD), e(T, P[5], C[3], ADD)     # C22

    limit = 20.0 + 2 * M * K * N / 2e9                 # a hang costs seconds, not the sweep
    rows = min(h, 32)                                   # checked sample: first rows of each half
    idx = torch.cat([torch.arange(rows), torch.arange(h, h + rows)])
    ref = X[idx, :K].double() @ W.double().T
    say("native.. ", end="")
    b.compile(M, native)
    us_n = min(b.run(limit) for _ in range(runs))
    got_n = torch.cat([b.read(b.out, (rows, N)), b.read(b.out + h * N * BPE, (rows, N))])
    say("\b\b\b\b\b\b\b\b\bstrassen.", end="")
    b.compile(h, strassen)
    us_s = min(b.run(limit) for _ in range(runs))
    say("\b\b\b\b\b\b\b\b\b", end="")
    blk = [b.read(a, (rows, n))[:, :n_real] for a in c]
    got_s = torch.cat([torch.cat([blk[0], blk[1]], 1), torch.cat([blk[2], blk[3]], 1)])
    flops = 2 * M * K * N
    return dict(native_ms=us_n / 1e3, strassen_ms=us_s / 1e3, ratio=us_s / us_n if us_n else float("nan"),
                native_gflops=flops / us_n / 1e3 if us_n else 0.0, strassen_gflops=flops / us_s / 1e3 if us_s else 0.0,
                native_snr_db=snr_db(ref, got_n), strassen_snr_db=snr_db(ref, got_s))


def sweep(args, Ms, Ks, Ns, path):
    done = set()
    if args.resume and os.path.exists(path):
        with open(path) as f:
            old = [complete(r) for r in csv.DictReader(f)]
        with open(path, "w", newline="") as f:          # same rows under the current header
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(old)
        done = {(int(r["M"]), int(r["K"]), int(r["N"])) for r in old}
        say(f"resume: {len(done)} shapes already in the CSV (failed ones are skipped too)")
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    Ms = [M for M in Ms if M % (1 << LEVEL) == 0]      # M must split LEVEL times
    todo = [(M, K, N) for K in Ks for N in Ns for M in Ms if (M, K, N) not in done]
    if not todo:
        say("nothing left to measure")
        return
    mM, mK, mN = max(Ms), max(Ks), max(Ns)
    b = Bench(args.dev, mM, mK, mN, args.engines)
    g = torch.Generator().manual_seed(args.seed)
    X = bf16(torch.randn(mM, mK, generator=g))
    Wsrc = torch.randn(mN, mK, generator=g)
    say(f"{len(todo)} shapes: M {Ms}\n          K {Ks}\n          N {Ns}")
    say("\ntotal GFLOP = matmul work executed (native -> Strassen, and the cut). GFLOPS = rate: native, then Strassen's\n"
        "real rate (its own work / its time) and effective rate (the native product's work / its time).")
    say(f"\n{'M':>5} {'K':>5} {'N':>5} | {'native ms':>10} {'strassen ms':>12} {'ratio':>6} | {'total GFLOP':>22} {'cut':>7} | "
        f"{'GFLOPS':>7} -> {'strassen':>20} | {'snr dB':>6} -> {'snr dB':>6}")
    t0, count, fails = time.time(), 0, 0
    for K in Ks:
        if not any(s[1] == K for s in todo):
            continue
        k, kp = K // 2, pad64(K // 2)
        widen = lambda t, rows, cols: torch.nn.functional.pad(t, (0, cols - t.shape[1], 0, rows - t.shape[0]))
        b.upload(b.x_full, X[:, :K])                    # row pitch K: any M <= mM is a prefix
        if LEVEL == 1:
            b.upload(b.xl, widen(X[:, :k], mM, kp))
            b.upload(b.xr, widen(X[:, k:K], mM, kp))
        for N in Ns:
            shapes = [s for s in todo if s[1] == K and s[2] == N]
            if not shapes:
                continue
            n, np_ = N // 2, pad64(N // 2)
            W = bf16(Wsrc[:N, :K] / K ** 0.5)
            Y11, Y12, Y21, Y22 = W[:n, :k], W[n:, :k], W[:n, k:], W[n:, k:]     # Yij = block (i, j) of W^T
            b.upload(b.w_full, W)
            if LEVEL == 1:
                for addr, combo in zip(b.w_c, [Y11 + Y22, Y11, Y12 - Y22, Y21 - Y11, Y22, Y11 + Y12, Y21 + Y22]):
                    b.upload(addr, widen(combo, np_, kp))
            else:           # the 7^LEVEL leaf matrices, formed in fp32 and rounded to bf16 once
                from strassen_levels import weight_leaves
                b.upload(b.lay.w, torch.stack(weight_leaves(widen(W, padl(N), padl(K)), LEVEL)))
            for M, _, _ in shapes:
                row = dict(M=M, K=K, N=N, peak_gflops=round(b.peak, 2))
                say(f"{M:5d} {K:5d} {N:5d} | ", end="")
                try:
                    res = measure(b, X, W, M, K, N, args.runs)
                    bad = any(v != v for v in (res["native_snr_db"], res["strassen_snr_db"]))
                    row.update({key: round(v, 4) for key, v in res.items()}, status="nan" if bad else "ok")
                    complete(row)
                    fails = 0
                    nat, st = work(M, K, N)
                    say(f"{res['native_ms']:10.3f} {res['strassen_ms']:12.3f} {res['ratio']:6.3f} | "
                        f"{nat / 1e9:9.3f} -> {st / 1e9:9.3f} {100 * (1 - st / nat):+6.1f}% | "
                        f"{res['native_gflops']:7.1f} -> {st / res['strassen_ms'] / 1e6 if res['strassen_ms'] else 0:5.1f} real, {res['strassen_gflops']:5.1f} eff | "
                        f"{res['native_snr_db']:6.1f} -> {res['strassen_snr_db']:6.1f}")
                except Exception as exc:                # one bad shape must not end the sweep
                    row["status"] = f"error: {type(exc).__name__}: {str(exc)[:120]}"
                    fails += 1
                    say(f"  <- stopped here | {row['status']}")
                with open(path, "a", newline="") as f:
                    csv.DictWriter(f, FIELDS).writerow(row)
                count += 1
                if fails:
                    if "imeout" in row["status"]:
                        say("stopping: a timed-out program leaves the engine stuck. Reset the board, then rerun with --resume")
                    else:
                        say("stopping: host-side error, nothing ran on the board for this shape; no reset needed. Rerun with --resume")
                    return
    say(f"\n{count} shapes in {time.time() - t0:.0f} s -> {path}")


# ----------------------------------------------------------------------------- chart

def plot(path, png, engines=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(path) as f:
        rows = [complete(r) for r in csv.DictReader(f) if r["status"] == "ok"]
    if not rows:
        say("no measured shapes to plot")
        return
    for r in rows:
        for key in FIELDS[:-1]:
            r[key] = float(r[key])
    Ms, Ks, Ns = (sorted({int(r[d]) for r in rows}) for d in "MKN")
    at = {(int(r["M"]), int(r["K"]), int(r["N"])): r for r in rows}
    peak = max(r["peak_gflops"] for r in rows)
    top = max(max(r["native_gflops"], r["strassen_gflops"]) for r in rows)
    ymax = max(top, peak) * 1.22

    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    NATIVE, STRASSEN = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRID, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(len(Ks), len(Ns), figsize=(max(4.4 * len(Ns), 12), max(2.9 * len(Ks), 5) + 1.6),
                             squeeze=False, sharey=True, facecolor=SURFACE)
    wins = 0
    for i, K in enumerate(Ks):
        for j, N in enumerate(Ns):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            xs = range(len(Ms))
            for x, M in zip(xs, Ms):
                r = at.get((M, K, N))
                if r is None:
                    continue
                ax.bar(x - 0.2, r["native_gflops"], 0.36, color=NATIVE, linewidth=0)
                ax.bar(x + 0.2, r["strassen_gflops"], 0.36, color=STRASSEN, linewidth=0)
                gain = (1 / r["ratio"] - 1) * 100 if r["ratio"] else 0.0
                wins += gain > 0
                ax.text(x, max(r["native_gflops"], r["strassen_gflops"]) + ymax * 0.015, f"{gain:+.0f}%",
                        ha="center", va="bottom", fontsize=7, color=INK if gain > 0 else MUTED,
                        fontweight="bold" if gain > 0 else "normal")
            if peak:
                ax.axhline(peak, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
            ax.set_ylim(0, ymax)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([str(M) for M in Ms], rotation=45 if len(Ms) > 8 else 0, fontsize=7)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"K = {K},  N = {N}", fontsize=9.5, color=INK, loc="left", pad=4)
            if j == 0:
                ax.set_ylabel("effective GFLOPS")
            if i == len(Ks) - 1:
                ax.set_xlabel("M (rows)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=NATIVE), plt.Rectangle((0, 0), 1, 1, color=STRASSEN),
               plt.Line2D([0], [0], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))]
    eng = "one engine" if engines == 1 else f"{engines} engines"
    labels = ["native matmul", f"Strassen ({LEVEL} level{'s' if LEVEL > 1 else ''}, {7 ** LEVEL} products)", f"peak, {eng} ({peak:.1f} GFLOPS)"]
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"Native matmul vs Strassen level {LEVEL} on {eng} (bf16)", x=0.005, y=0.995, ha="left", va="top",
                 fontsize=17, fontweight="bold")
    fig.text(0.005, 0.972 - 0.2 / fig.get_figheight(),
             f"Effective GFLOPS = 2MKN / time, so a taller bar is a faster matmul. The number over each pair is Strassen's speed-up "
             f"over native. Strassen is faster on {wins} of {len(rows)} shapes. Output stitching is not timed.",
             ha="left", va="top", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.25 / fig.get_figheight()))
    fig.savefig(png, dpi=130, facecolor=SURFACE)
    say(f"chart -> {png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--engines", type=int, default=1, help="engines to shard both versions across (rows); results go to *_e<N>")
    ap.add_argument("--strassen-level", type=int, choices=(1, 2, 3), default=1,
                    help="Strassen recursion depth: 7, 49 or 343 products; results go to *_L<n> for 2 and 3")
    # with --engines: level 1 shards half-rows; levels 2-3 give each engine a row range and a full recursion
    ap.add_argument("--grid", choices=sorted(GRIDS), default="pow2")
    for d in "mkn":
        ap.add_argument(f"--{d}", default=None, help=f"comma-separated {d.upper()} values (overrides --grid)")
    ap.add_argument("--max-m", type=int, default=4096)
    ap.add_argument("--max-k", type=int, default=8064)
    ap.add_argument("--max-n", type=int, default=8192)
    ap.add_argument("--runs", type=int, default=1, help="timed runs per program; the fastest is kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "strassen_sweep"), help="output path without extension")
    ap.add_argument("--resume", action="store_true", help="keep the CSV and measure only the shapes it lacks")
    ap.add_argument("--plot-only", action="store_true", help="redraw the chart from the CSV; no device")
    args = ap.parse_args()
    global LEVEL
    LEVEL = args.strassen_level

    lists = {}
    for d, cap in (("M", args.max_m), ("K", args.max_k), ("N", args.max_n)):
        given = getattr(args, d.lower())
        vals = sorted({int(v) for v in given.split(",")}) if given else GRIDS[args.grid][d]
        lists[d] = [v for v in vals if v <= cap]
        assert lists[d], f"no {d} values at or below {cap}"
    assert all(M % 2 == 0 for M in lists["M"]), "every M must be even"
    assert all(v % 64 == 0 for v in lists["K"] + lists["N"]), "every K and N must be a multiple of 64"
    assert max(lists["K"]) < 8192, "K >= 8192 takes matmat_mul_core's 16-column fallback; keep K <= 8064"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if LEVEL > 1:
        args.out += f"_L{LEVEL}"
    if args.engines > 1:
        args.out += f"_e{args.engines}"
    path, png = args.out + ".csv", args.out + ".png"
    if not args.plot_only:
        sweep(args, lists["M"], lists["K"], lists["N"], path)
    plot(path, png, args.engines)


if __name__ == "__main__":
    main()
