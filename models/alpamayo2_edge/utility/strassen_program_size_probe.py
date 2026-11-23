#!/usr/bin/env python3
"""Do the smaller Strassen programs compute exactly what the inline program computes, and as fast?

nn_lib section 10 can emit a Strassen matmul in fewer instructions (PROGRAM SIZE in its header).
This probe runs each way against the inline emission (every op in line, whatever nn_lib's
defaults), on the same data, on every engine, on the Alpamayo prefill shapes (rows per engine x
K -> N), and wants the SAME BITS back:

  ops     nn_lib.STRASSEN_SHARE_OPS = 1: every distinct product / elementwise pass of a call is
          emitted once and jumped to, its addresses in registers.    checked: the matmul output
  pitch   out_pitch=: the result written into a column slice of a wider buffer.   checked: the
          slice equals the contiguous result, the columns outside it are untouched
  shared  shared=d: --layers layers (different weights, same shape) through ONE emitted body per
          engine, against the same layers as separate literal calls. checked: every layer's output
  sums    nn_lib.STRASSEN_SUMS_LOOPED = 1: the pieces of a weight sum in a hardware loop.
                                                                     checked: every sum buffer

Per program: instructions and bytes (engine 0, and bytes over all engines), board time (fastest
of --runs), and for a new mode FASTER / same / SLOWER than the old one. RESULT: PASS needs every
new program bit-identical to the old one, no larger, and at most 1% slower.

The tests run in the order above; `sums` and `shared` run their 4-bit cases (an on-chip
dequantize addressed from a register) last. A "running: ..." line precedes every program. The probe stops at the first timeout: the engine
is then stuck, and the board needs a reset before anything else is run.

  python models/alpamayo2_edge/utility/strassen_program_size_probe.py                # six cases, a few minutes
  python models/alpamayo2_edge/utility/strassen_program_size_probe.py --tests ops pitch
  python models/alpamayo2_edge/utility/strassen_program_size_probe.py --tests shared sums --weights bf16
  python models/alpamayo2_edge/utility/strassen_program_size_probe.py --full         # 2 weights x 2 levels x 2 activations x 3 shapes
  python models/alpamayo2_edge/utility/strassen_program_size_probe.py --engines 1 --shape 408 2048 2048 --levels 1
"""
import argparse
import contextlib
import sys
import traceback

import torch

from strassen_probe_util import ACT_FN, ACT_KW, BF, FP, open_board, put as put_on, snr

TESTS = ("ops", "pitch", "shared", "sums")
SHAPES = [(408, 2048, 2048), (408, 2048, 4608), (408, 4608, 2048)]     # q / o, fc1 (ReLU in the model), fc2
# the default: every weight kind, level, activation and shape at least once (weights, level, shape, activation, tests)
QUICK = [("if4", 1, 0, None, ("ops", "pitch", "shared", "sums")), ("if4", 2, 1, "relu", ("ops", "shared", "sums")),
         ("if4", 2, 2, None, ("ops", "sums")), ("bf16", 1, 1, "relu", ("ops", "shared", "sums")),
         ("bf16", 2, 0, None, ("ops", "pitch", "sums")), ("bf16", 2, 2, "relu", ("ops", "sums"))]
PAD, COL0 = 256, 128            # pitch test: the wide buffer has PAD more columns, the result starts at column COL0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="max", help="engine count, or 'max' = every core the bitstream reports")
    ap.add_argument("--tests", nargs="+", choices=TESTS, default=list(TESTS), help="which tests to run (default: all four)")
    ap.add_argument("--weights", nargs="+", choices=("if4", "bf16"), default=None)
    ap.add_argument("--levels", nargs="+", type=int, choices=(1, 2), default=None)
    ap.add_argument("--act", nargs="+", choices=("none", "relu", "gelu", "silu"), default=None)
    ap.add_argument("--shape", nargs=3, type=int, action="append", metavar=("ROWS", "K", "N"),
                    help="rows per engine, input width, output width; repeatable (default: the three prefill shapes)")
    ap.add_argument("--full", action="store_true",
                    help="every weights x level x activation x shape with every test (default: six cases covering each value)")
    ap.add_argument("--layers", type=int, default=3, help="layers the shared test runs through one body")
    ap.add_argument("--runs", type=int, default=3, help="executions per program; the time reported is the fastest")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds one program may take")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.full or args.weights or args.levels or args.act or args.shape:
        shapes = [tuple(s) for s in args.shape] if args.shape else SHAPES
        cases = [(w, L, r, K, N, None if a == "none" else a, TESTS) for w in (args.weights or ("if4", "bf16"))
                 for L in (args.levels or (1, 2)) for r, K, N in shapes for a in (args.act or ("none", "relu"))]
    else:
        cases = [(w, L, *SHAPES[s], a, tests) for w, L, s, a, tests in QUICK]
    assert all(r % (1 << L) == 0 and K % (64 << L) == 0 and N % (64 << L) == 0 for _, L, r, K, N, _, _ in cases), \
        "a case needs rows a multiple of 2^level and K, N multiples of 64 * 2^level"
    assert args.layers >= 2 and args.runs >= 1

    T, ue, NE, cores = open_board(args.dev, args.engines)
    import nn_lib
    print(f"engines: {NE} of {cores}; rows are per engine; nn_lib defaults: STRASSEN_SHARE_OPS = {nn_lib.STRASSEN_SHARE_OPS}, "
          f"STRASSEN_SUMS_LOOPED = {nn_lib.STRASSEN_SUMS_LOOPED}")

    # the kernel's private scratch, one zero-filled copy per engine and depth, sized for the largest case
    depth = max(c[1] for c in cases)
    for d in range(depth):
        sizes = [nn_lib.strassen_scratch_elements(r, K, N, L)[d] for _, L, r, K, N, _, _ in cases if L > d]
        ue._palloc(f"ZX{d}", max(s[0] for s in sizes))
        ue._palloc(f"ZP{d}", max(s[1] for s in sizes))
    t_base, p_base = ue.get_tensor_dram_addr(), ue.get_params_dram_addr()     # every case reuses both arenas from here

    put = lambda t: put_on(ue, t)

    def bits(addr, shape, dtype=None):
        return ue.dma_from_accelerator_memory(addr, shape, dtype=dtype).view(torch.int16)

    def poison(addr, elements):                                             # NaN in bf16 and in fp16: a value nothing may leave behind
        ue.dma_to_accelerator_memory(addr, torch.full((elements,), float("nan"), dtype=BF))

    def move(e_ue, src, dst, elements, hop=131072):                         # DRAM -> DRAM through URAM A
        for e0 in range(0, elements, hop):
            take = min(hop, elements - e0)
            e_ue.accelerator_memory_to_sram(accelerator_dram_address=src + e0 * 2, sram_address=0x00000, element_size=take)
            e_ue.sram_to_accelerator_memory(sram_address=0x00000, accelerator_dram_address=dst + e0 * 2, element_size=take)

    @contextlib.contextmanager
    def switches(share_ops=0, sums_looped=0):                               # "old" is always both at 0, whatever nn_lib's defaults
        keep = nn_lib.STRASSEN_SHARE_OPS, nn_lib.STRASSEN_SUMS_LOOPED
        nn_lib.STRASSEN_SHARE_OPS, nn_lib.STRASSEN_SUMS_LOOPED = share_ops, sums_looped
        try:
            yield
        finally:
            nn_lib.STRASSEN_SHARE_OPS, nn_lib.STRASSEN_SUMS_LOOPED = keep

    count, now = [0], ["start-up"]
    checks, table = [], []

    def check(what, ok, detail):
        checks.append((what, bool(ok), detail))
        if not ok:
            print(f"    CHECK FAILED: {what}: {detail}", flush=True)

    def measure(label, rounds, prepare, read, **sw):
        """One program of the given rounds (each a body(E) run by every engine): compile it with
        the switches sw, prepare() the buffers, run it, read() the result, then run it again for
        the time. -> (fastest board us, [program bytes per engine], what read() returned)."""
        count[0] += 1
        name, now[0] = f"probe{count[0]}", f"compiling: {label}"

        def emit():
            for body in rounds:
                def fenced(E, body=body):
                    E.ue.reset_isa_reg_counter()
                    body(E)
                ue._round(fenced)
        workers = ue.sched.workers if NE > 1 else []
        before = [ue.get_program_dram_addr()] + [w.get_program_dram_usage() for w in workers]
        with switches(**sw):
            ue._compile(name, emit, parallel=True)
        size = [ue.get_program_dram_addr() - before[0]] + [w.get_program_dram_usage() - b for w, b in zip(workers, before[1:])]
        prepare()
        us, got = None, None
        for i in range(args.runs):
            now[0] = label if i == 0 else f"{label} (run {i + 1})"
            print(f"  running: {now[0]} ...", flush=True)                   # if the board hangs, this names the culprit
            t = ue._exec(name, args.timeout)
            us = t if us is None else min(us, t)
            if i == 0:
                got = read()
        return us, size, got

    def row(tag, label, us, size, old=None):
        """A line of the results table; for a new mode, its time and size against the old program's."""
        line = f"    {label:46s} {size[0] // 32:8d} instr {size[0]:10d} B {sum(size):11d} B all {us / 1e3:10.2f} ms"
        if old is not None:
            d = (us - old[0]) / old[0]
            word = "FASTER" if d < -0.005 else "SLOWER" if d > 0.01 else "same"
            line += f"  {100 * d:+6.2f}% {word}, {sum(size) / sum(old[1]):.2f}x the bytes"
            check(f"{tag}: {label} is not slower", d <= 0.01, f"{us / 1e3:.2f} ms against {old[0] / 1e3:.2f} ms ({100 * d:+.2f}%)")
            check(f"{tag}: {label} is not larger", sum(size) <= sum(old[1]), f"{sum(size)} B against {sum(old[1])} B")
        print(line, flush=True)
        table.append((tag, line))

    def same_bits(tag, what, got, ref):
        diff = int((got != ref).sum())
        check(f"{tag}: {what} bit-identical", diff == 0, f"{diff} of {ref.numel()} values differ")
        return diff == 0

    def setup(kind, L, r, K, N, layers=1, x=True):
        """Fresh data for a case: X, the sum buffers, and per layer its weight as cells and its tree.
        Every layer stores its cells (and bias shares) in the same order, so they lie the same way."""
        ue._tensor_dram_addr, ue._next_params_dram_addr = t_base, p_base
        g = torch.Generator().manual_seed(args.seed)
        n, k, M = N >> L, K >> L, r * NE
        X = torch.randn(M, K, generator=g).to(BF)
        D = dict(X=X, a_x=put(X) if x else None, a_eye=put(torch.eye(64, dtype=BF)), n=n, k=k, M=M,
                 a_s=[ue._alloc(n * k) for _ in range(nn_lib.strassen_online_sum_count(L))], layers=[])
        qstore = lambda d, s_, rows_, cols, t=None: ue._qstore((d, s_, rows_, cols))
        for _ in range(layers):
            W = torch.randn(N, K, generator=g) / K ** 0.5
            if kind == "if4":
                data, scales, _, _ = T.if4_pack(W)
                bias, cells = None, nn_lib.strassen_if4_cells(data, scales, N, K, L, qstore)
                tree = nn_lib.strassen_online_tree(cells, D["a_s"], L)
            else:
                bias, cells = torch.randn(N, generator=g).to(BF).float(), nn_lib.strassen_bf16_cells(W, L, put)
                tree = nn_lib.strassen_online_tree(cells, D["a_s"], L, bias, put)
            D["layers"].append(dict(W=W, bias=bias, cells=cells, tree=tree))
        assert ue.get_params_dram_addr() <= ue.tensor_base, "the probe's weights ran into the tensor arena"
        items = nn_lib.strassen_online_items(n, k, L)
        D["striped"] = [items[e::NE] for e in range(NE)]                    # how the model deals them: every NE-th piece
        D["blocks"] = [items[e * len(items) // NE:(e + 1) * len(items) // NE] for e in range(NE)]   # runs of neighbouring pieces
        return D

    def tests_of(case):
        kind, L, r, K, N, act, _ = case
        tag = f"{kind} L{L} {r}x{K}->{N}{' ' + act if act else ''}"
        kw = ACT_KW[act]
        rows_of = lambda E: E.e * r                                         # this engine's first row
        scratch = lambda E: [(ue.pbuf[E.e][f"ZX{d}"], ue.pbuf[E.e][f"ZP{d}"]) for d in range(L)]
        sizes = nn_lib.strassen_scratch_elements(r, K, N, L)

        def clean(*outs):
            """Before a program whose bits are compared: zeroed scratch (as a model starts), poisoned outputs."""
            def prepare():
                for e in range(NE):
                    for d in range(L):
                        ue._zero(ue.pbuf[e][f"ZX{d}"], sizes[d][0])
                        ue._zero(ue.pbuf[e][f"ZP{d}"], sizes[d][1])
                for addr, elements in outs:
                    poison(addr, elements)
            return prepare

        def matmul(D, layer, out, pitch=None, **more):
            p = N if pitch is None else pitch
            return lambda E: nn_lib.strassen_matmul_core_dram(
                E.ue, r, K, N, D["a_x"] + rows_of(E) * K * 2, layer["tree"], out + rows_of(E) * p * 2, scratch(E),
                activation=kw or None, identity_addr=D["a_eye"] if kw else None, **({} if pitch is None else {"out_pitch": pitch}), **more)

        def sums(D, layer, deal, **more):
            return lambda E: nn_lib.strassen_online_sums_core(E.ue, layer["cells"], D["n"], D["k"], D["a_s"], L, deal[E.e], **more)

        def read_sums(D):
            return torch.stack([bits(a, (D["n"] * D["k"],), FP) for a in D["a_s"]])

        def sane(D, layer, C, what):
            """The old program's result is the layer's product (so identical bits mean something)."""
            want = ACT_FN[act](D["X"].float() @ layer["W"].to(BF).float().T + (0 if layer["bias"] is None else layer["bias"]))
            got, need = C.view(BF).float(), 10 if kind == "if4" else 35      # 4-bit weights against the unquantized ones
            ok = bool(torch.isfinite(got).all()) and snr(want, got) >= need
            check(f"{tag}: {what} is the layer's product", ok, f"{snr(want, got):.1f} dB against the host (needs {need})")

        def ops():
            D = setup(kind, L, r, K, N)
            layer, M = D["layers"][0], D["M"]
            a_o = ue._alloc(M * N)
            measure(f"{tag}: sums, old", [sums(D, layer, D["striped"])], lambda: None, lambda: None)
            old = measure(f"{tag}: matmul, old", [matmul(D, layer, a_o)], clean((a_o, M * N)), lambda: bits(a_o, (M, N)))
            new = measure(f"{tag}: matmul, STRASSEN_SHARE_OPS = 1", [matmul(D, layer, a_o)], clean((a_o, M * N)),
                          lambda: bits(a_o, (M, N)), share_ops=1)
            sane(D, layer, old[2], "the old matmul")
            same_bits(tag, "matmul with STRASSEN_SHARE_OPS = 1", new[2], old[2])
            row(tag, "matmul, old", *old[:2])
            row(tag, "matmul, STRASSEN_SHARE_OPS = 1", *new[:2], old=old)

        def pitch():
            D = setup(kind, L, r, K, N)
            layer, M, wide = D["layers"][0], D["M"], N + PAD
            a_o, a_w = ue._alloc(M * N), ue._alloc(M * wide)
            fence = torch.randn(M, wide, generator=torch.Generator().manual_seed(args.seed + 1)).to(BF)
            measure(f"{tag}: sums, old", [sums(D, layer, D["striped"])], lambda: None, lambda: None)
            old = measure(f"{tag}: matmul, old, contiguous output", [matmul(D, layer, a_o)], clean((a_o, M * N)),
                          lambda: bits(a_o, (M, N)))

            def prepare():
                clean()()
                ue.dma_to_accelerator_memory(a_w, fence.flatten())
            new = measure(f"{tag}: matmul, old, out_pitch = {wide} from column {COL0}", [matmul(D, layer, a_w + COL0 * 2, wide)],
                          prepare, lambda: bits(a_w, (M, wide)))
            sane(D, layer, old[2], "the contiguous matmul")
            same_bits(tag, f"columns {COL0}..{COL0 + N} of the wide buffer", new[2][:, COL0:COL0 + N], old[2])
            outside = torch.ones(wide, dtype=torch.bool)
            outside[COL0:COL0 + N] = False
            same_bits(tag, "the columns outside the slice (untouched)", new[2][:, outside], fence.view(torch.int16)[:, outside])
            row(tag, "matmul, contiguous output", *old[:2])
            row(tag, f"matmul, out_pitch = {wide}", *new[:2], old=old)

        def shared():
            D = setup(kind, L, r, K, N, layers=args.layers)
            M = D["M"]
            a_o, res = ue._alloc(M * N), [ue._alloc(M * N) for _ in D["layers"]]

            def rounds(**more):
                """Per layer: its sums, then (next round) its matmul into the one output buffer and a copy of that out."""
                out = []
                for i, layer in enumerate(D["layers"]):
                    out.append(sums(D, layer, D["striped"], **more))
                    out.append(lambda E, i=i, layer=layer: (
                        matmul(D, layer, a_o, **more)(E),
                        move(E.ue, a_o + rows_of(E) * N * 2, res[i] + rows_of(E) * N * 2, r * N)))
                return out
            read = lambda: [bits(a, (M, N)) for a in res]
            prepare = clean((a_o, M * N), *[(a, M * N) for a in res])
            old = measure(f"{tag}: {args.layers} layers, a literal call each", rounds(), prepare, read)
            kept = {}
            new = measure(f"{tag}: {args.layers} layers, shared=", rounds(shared=kept), prepare, read)
            sane(D, D["layers"][-1], old[2][-1], "the last layer's literal matmul")
            check(f"{tag}: the layers differ (the test can tell them apart)", bool((old[2][0] != old[2][1]).any()), "layers 0 and 1 gave the same output")
            for i in range(args.layers):
                same_bits(tag, f"layer {i} through the shared body", new[2][i], old[2][i])
            want = NE + sum(1 for e in range(NE) if D["striped"][e])        # one matmul body per engine, one sums body where it has pieces
            calls = sorted({v["uses"] for v in kept.values()})
            check(f"{tag}: one body per engine and kind, each run by every layer", len(kept) == want and calls == [args.layers],
                  f"{len(kept)} bodies (expected {want}), calls per body {calls} (expected [{args.layers}])")
            row(tag, f"{args.layers} layers, a literal call each", *old[:2])
            row(tag, f"{args.layers} layers, shared= ({len(kept)} bodies)", *new[:2], old=old)

        def looped_sums():
            D = setup(kind, L, r, K, N, x=False)
            layer, total = D["layers"][0], len(D["a_s"]) * D["n"] * D["k"]
            prepare = lambda: [poison(a, D["n"] * D["k"]) for a in D["a_s"]]
            old = measure(f"{tag}: sums, old", [sums(D, layer, D["blocks"])], prepare, lambda: read_sums(D))
            new = measure(f"{tag}: sums, STRASSEN_SUMS_LOOPED = 1", [sums(D, layer, D["blocks"])], prepare, lambda: read_sums(D),
                          sums_looped=1)
            nan = int((old[2].view(FP) != old[2].view(FP)).sum())
            check(f"{tag}: the old sums fill every buffer", nan == 0, f"{nan} of {total} values are still the poison")
            same_bits(tag, "sums with STRASSEN_SUMS_LOOPED = 1", new[2], old[2])
            row(tag, f"sums, old ({len(D['a_s'])} buffers)", *old[:2])
            row(tag, "sums, STRASSEN_SUMS_LOOPED = 1", *new[:2], old=old)

        return {"ops": ops, "pitch": pitch, "shared": shared, "sums": looped_sums}

    crashed, done = None, set()
    try:
        for test in [t for t in TESTS if t in args.tests]:
            todo = [c for c in cases if test in c[6]]
            if test in ("shared", "sums"):                                  # the register-addressed dequantize (4-bit) last
                todo.sort(key=lambda c: c[0] == "if4")
            for case in todo:
                kind, L, r, K, N, act, _ = case
                key = (test, kind, L, K, N) if test == "sums" else (test, kind, L, r, K, N, act)    # the sums do not depend on rows / activation
                if key in done:
                    continue
                done.add(key)
                print(f"\n[{test}] {kind}, level {L}, {r} rows/engine x {K} -> {N}"
                      + (f", {act}" if act and test != "sums" else ""), flush=True)
                tests_of(case)[test]()
    except Exception as error:
        traceback.print_exc()
        crashed = (now[0], isinstance(error, TimeoutError))

    print(f"\nresults, {NE} engine{'s' if NE > 1 else ''} (program of engine 0; bytes of all engines; fastest of {args.runs} runs)")
    last = None
    for tag, line in table:
        if tag != last:
            print(f"  {tag}")
        last = tag
        print(line)
    bad = [c for c in checks if not c[1]]
    print(f"\nchecks: {len(checks) - len(bad)} of {len(checks)} passed")
    for what, _, detail in bad:
        print(f"  failed: {what}: {detail}")
    if crashed:
        why = "TIMED OUT: the engine is stuck, reset the board before running anything else" if crashed[1] \
            else "raised the error above; if it was a timeout the board needs a reset"
        print(f"RESULT: FAIL (stopped in '{crashed[0]}': {why})")
        return 1
    print("RESULT: " + ("PASS" if checks and not bad else f"FAIL ({len(bad)} check{'s' if len(bad) != 1 else ''} failed)" if bad
                        else "FAIL (nothing was run)"))
    return 0 if checks and not bad else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL (the probe stopped with the error above)")
        code = 1
    sys.exit(code)
