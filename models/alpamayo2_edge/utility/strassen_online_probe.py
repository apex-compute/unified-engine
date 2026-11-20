#!/usr/bin/env python3
"""Strassen WITHOUT storing expanded weights ("online leaves"), levels 1 and 2, on 4-bit (IF4) and
on bf16 layers, on every engine.

C = X . W^T (+ bias on a bf16 layer) with the rows split evenly over the engines, as:

  plain              the ordinary matmul                                    (what models run today)
  stored leaves L    the leaf weights are made on the host and stored:
                       IF4 layer   nn_lib.strassen_weight_tree_if4 (blocks IF4, sums IF8): 2.9x / 5.7x
                       bf16 layer  nn_lib.strassen_weight_tree (fp16 leaves):              1.75x / 3.06x
  online leaves L    the weight is stored ONCE as its 4^L cells (1.00x). Before the matmul the
                     engines make every leaf that is a sum on chip and write it to a shared
                     scratch (nn_lib.strassen_online_sums_core), then run the 7^L products.

Per case: the time of each against the plain matmul, the SNR, the weight bytes, and what making
the sums costs on its own. The last lines are a summary and RESULT: PASS / FAIL. PASS means every
number is CORRECT: the sums the board made match the host, and every Strassen reproduces the plain
matmul to the accuracy expected of its level. Which variant is fastest is the summary's verdict.

  python models/alpamayo2_edge/utility/strassen_online_probe.py                       # every engine, default cases
  python models/alpamayo2_edge/utility/strassen_online_probe.py --engines 1
  python models/alpamayo2_edge/utility/strassen_online_probe.py --case if4 832 4608 2048 --case bf16 720 4352 1152
"""
import argparse
import os
import sys
import traceback

import numpy as np
import torch

BF, FP = torch.bfloat16, torch.float16
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (weights, rows PER ENGINE, K, N)
CASES = [("if4", 408, 2048, 2048), ("if4", 408, 4608, 2048), ("if4", 832, 4608, 2048),
         ("bf16", 408, 4608, 2048), ("bf16", 720, 4352, 1152)]


def snr(ref, x):
    ref, x = ref.double(), x.double()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-300)))


def decode(data, scales, N, K):
    """An IF4 pack -> code x |scale| in float64, no rounding: the sign of a block's scale says
    whether its 64 codes are integers -8..7 or the small-float table. (quant_lib.dequant multiplies
    in bf16 and is off by up to a bf16 step, so it cannot check an fp16 result.)"""
    nib = np.frombuffer(data, dtype=np.uint8).reshape(N, K // 2)
    codes = np.stack([nib & 0xF, nib >> 4], -1).reshape(N, K // 64, 64)
    sc = torch.frombuffer(bytearray(scales), dtype=torch.int16).view(BF).double().numpy().reshape(N, K // 64, 1)
    table = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    ints = np.where(codes >= 8, codes.astype(np.int64) - 16, codes).astype(np.float64)
    return torch.from_numpy((np.where(sc < 0, ints, table[codes]) * np.abs(sc)).reshape(N, K))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="max", help="engine count, or 'max' = every core the bitstream reports")
    ap.add_argument("--case", nargs=4, action="append", metavar=("WEIGHTS", "ROWS", "K", "N"),
                    help="if4 | bf16, rows per engine, input width, output width; repeatable")
    ap.add_argument("--levels", type=int, default=2, choices=(1, 2))
    ap.add_argument("--act", choices=["gelu", "silu", "relu"], default=None,
                    help="fuse this activation in the plain matmul and in the Strassen kernel (activation= / identity_addr=)")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cases = [(c[0], int(c[1]), int(c[2]), int(c[3])) for c in args.case] if args.case else CASES
    assert all(c[0] in ("if4", "bf16") and c[1] % 2 == 0 and c[2] % 128 == 0 and c[3] % 128 == 0 for c in cases), \
        "a case is: if4 | bf16, even rows, K and N multiples of 128"

    sys.path.insert(0, HERE)
    import alpamayo2_edge_test as T
    import nn_lib
    import user_dma_core
    from user_dma_core import TYPE
    dev = args.dev or T.CFG["defaults"]["dev"]
    user_dma_core.set_dma_device(dev)
    user_dma_core.configure_clock_from_hardware()
    cores = getattr(user_dma_core, "ANDROMEDA_CORE_COUNT", None) or 1
    NE = cores if args.engines == "max" else int(args.engines)
    assert 1 <= NE <= cores, f"--engines {NE} but the bitstream reports {cores} core(s)"
    ue = T.Alpamayo2Edge_UnifiedEngine(dev=dev, engines=NE)
    base = ue.get_tensor_dram_addr()
    print(f"engines: {NE} of {cores}; rows are per engine")

    def levels_of(r, K, N):
        return [L for L in range(1, args.levels + 1) if r % (1 << L) == 0 and K % (64 << L) == 0 and N % (64 << L) == 0]

    # the kernel's private scratch, one zero-filled copy per engine and depth, sized for the largest case
    for d in range(args.levels):
        sizes = [nn_lib.strassen_scratch_elements(r, K, N, max(levels_of(r, K, N)))[d]
                 for _, r, K, N in cases if len(levels_of(r, K, N)) > d]
        if sizes:
            ue._palloc(f"OX{d}", max(s[0] for s in sizes))
            ue._palloc(f"OP{d}", max(s[1] for s in sizes))

    def put(t):                                                             # bf16 or fp16, uploaded as it is
        a = ue._alloc(t.numel())
        ue.dma_to_accelerator_memory(a, t.contiguous().flatten())
        return a

    def out_buffer(elements):
        a = ue.allocate_tensor_dram(elements * 2, align_bytes=128)
        assert ue.get_tensor_dram_addr() <= ue.tensor_limit, "tensor arena ran into the program arena"
        return a

    def reg(E, value):
        r = E.ue.alloc_isa_reg()
        E.ue.generate_instruction_add_set(r, value)
        return r

    count = [0]

    def timed(*rounds):
        """One program of the given rounds (each a body(E) run by every engine) -> board time, us."""
        count[0] += 1

        def emit():
            for body in rounds:
                def fenced(E, body=body):
                    E.ue.reset_isa_reg_counter()
                    body(E)
                ue._round(fenced)
        ue._compile(f"probe{count[0]}", emit, parallel=True)
        return ue._exec(f"probe{count[0]}")

    checks, summary = [], []
    act = {"gelu": {"gelu_enable": True}, "silu": {"silu_enable": True},
           "relu": {"clamp_enable": True, "clamp_min": 0.0, "clamp_max": float("inf")}, None: {}}[args.act]
    fn = {"gelu": lambda v: v * torch.sigmoid(1.702 * v), "silu": lambda v: v * torch.sigmoid(v),
          "relu": lambda v: v.clamp(min=0), None: lambda v: v}[args.act]
    if args.act:
        print(f"activation: {args.act}, fused in the plain matmul and in the kernel")

    def check(what, ok, detail):
        checks.append((what, bool(ok), detail))
        if not ok:
            print(f"    CHECK FAILED: {what}: {detail}", flush=True)

    for kind, r, K, N in cases:
        M, Ls = r * NE, levels_of(r, K, N)
        name = f"{kind} {r} rows/engine x {K} -> {N}"
        ue._tensor_dram_addr = base
        g = torch.Generator().manual_seed(args.seed)
        X = torch.randn(M, K, generator=g).to(BF)
        W = torch.randn(N, K, generator=g) / K ** 0.5
        print(f"\n{name}  ({M} rows in all; levels {Ls})", flush=True)
        a_x = put(X)
        a_eye = put(torch.eye(64, dtype=BF)) if act else None
        rows_of = lambda E: (a_x + E.e * r * K * 2, E.e * r * N * 2)        # this engine's X rows, and its output offset
        scratch = lambda E: [(ue.pbuf[E.e][f"OX{d}"], ue.pbuf[E.e][f"OP{d}"]) for d in range(max(Ls))]
        products = lambda tree, out: (lambda E: nn_lib.strassen_matmul_core_dram(
            E.ue, r, K, N, rows_of(E)[0], tree, out + rows_of(E)[1], scratch(E), activation=act or None, identity_addr=a_eye))
        qstore = lambda d, s_, rows, cols, t=None: ue._qstore((d, s_, rows, cols))

        # ---- the plain matmul, and what "correct" means for this layer
        a_c = out_buffer(M * N)
        if kind == "if4":
            data, scales, _, _ = pack = T.if4_pack(W)
            size, bias = len(data) + len(scales), None
            a_w = ue._qstore(pack)
            t0 = timed(lambda E: E.ue.matmat_mul_core(M=r, K=K, N=N, A_DRAM_ADDR=rows_of(E)[0], B_DRAM_ADDR=a_w[0],
                                                      OUTPUT_DRAM_ADDR=a_c + rows_of(E)[1], gpr_M_reg=reg(E, r),
                                                      is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=a_w[1], **act))
            truth = fn(X.double() @ W.to(BF).double().T)                      # the product with 16-bit weights
            exact = decode(data, scales, N, K)
            loaded = lambda t: t.to(FP).double()                            # a cell is dequantized to fp16 on the way in
            need = lambda L, online: 40 if online else (30, 28)[L - 1]      # agreement with the plain IF4 result, dB
        else:
            bias = torch.randn(N, generator=g).to(BF).float()
            size = N * K * 2
            a_w, a_b = put(W.to(BF)), put(bias.to(BF))
            t0 = timed(lambda E: E.ue.matmat_mul_core(M=r, K=K, N=N, A_DRAM_ADDR=rows_of(E)[0], B_DRAM_ADDR=a_w,
                                                      OUTPUT_DRAM_ADDR=a_c + rows_of(E)[1], C_DRAM_ADDR=a_b,
                                                      bias_mode="broadcast_N", gpr_M_reg=reg(E, r), **act))
            truth = fn(X.double() @ W.to(BF).double().T + bias.double())      # the exact product
            exact = W.to(BF).double()
            loaded = lambda t: t
            need = lambda L, online: (50, 47)[L - 1]                        # SNR against the exact product, dB
        plain = ue.dma_from_accelerator_memory(a_c, (M, N)).double()
        s0 = snr(truth, plain)

        rows, results = [("plain", t0, plain, 1.0, None)], {}
        mark = ue.get_tensor_dram_addr()
        for L in Ls:
            n, k, cg = N >> L, K >> L, 1 << L
            # ---- stored leaves
            ue._tensor_dram_addr = mark
            nbytes = [0]
            if kind == "if4":
                def store_pack(d, s_, rows_, cols, t):
                    nbytes[0] += len(d) + len(s_)
                    return ue._qstore((d, s_, rows_, cols))
                tree = nn_lib.strassen_weight_tree_if4(data, scales, N, K, None, L, store_pack)
            else:
                def store(t):
                    nbytes[0] += t.numel() * 2 if t.dim() == 2 else 0
                    return put(t)
                tree = nn_lib.strassen_weight_tree(W.to(BF).float(), bias, L, store, store)
            a_o = out_buffer(M * N)
            t_st = timed(products(tree, a_o))
            C_st = ue.dma_from_accelerator_memory(a_o, (M, N)).double()
            rows.append((f"stored leaves, level {L}", t_st, C_st, nbytes[0] / size, None))

            # ---- online leaves: the weight stored once as its cells, the sums made on chip
            ue._tensor_dram_addr = mark
            cells = (nn_lib.strassen_if4_cells(data, scales, N, K, L, qstore) if kind == "if4"
                     else nn_lib.strassen_bf16_cells(W, L, put))
            a_s = [ue._alloc(n * k) for _ in range(nn_lib.strassen_online_sum_count(L))]
            tree = nn_lib.strassen_online_tree(cells, a_s, L, bias, put if bias is not None else None)
            items = nn_lib.strassen_online_items(n, k, L)
            deal = [items[e::NE] for e in range(NE)]
            make = lambda E: nn_lib.strassen_online_sums_core(E.ue, cells, n, k, a_s, L, deal[E.e])
            a_o = out_buffer(M * N)
            t_prep = timed(make)
            t_on = timed(make, products(tree, a_o))
            C_on = ue.dma_from_accelerator_memory(a_o, (M, N)).double()
            rows.append((f"online leaves, level {L}", t_on, C_on, 1.0, (t_prep, len(items))))

            # the sums the board made, against the host doing the same adds in fp16
            cell = {(i, j): loaded(exact[i * n:(i + 1) * n, j * k:(j + 1) * k]) for i in range(cg) for j in range(cg)}
            recipes = [t for t in nn_lib.strassen_leaf_cells(L) if len(t) > 1]
            worst, same, differ, shown = 0.0, 0, 0, []
            for terms, addr in zip(recipes, a_s):
                got = ue.dma_from_accelerator_memory(addr, (n, k), dtype=FP).double()
                exp, step = cell[terms[0][0]], cell[terms[0][0]].abs()
                for c, sign in terms[1:]:
                    exp, step = (exp + sign * cell[c]).to(FP).double(), step + cell[c].abs()
                # one fp16 step of the operands, or the smallest fp16 number for values near zero
                # (down there fp16 is a fixed grid of 2^-24, and bf16 weights can fall between its points)
                worst = max(worst, float(((got - exp).abs() / (step * 2 ** -10).clamp_min(2 ** -24)).max()))
                same += int((got == exp).sum())
                if bool((got != exp).any()) and len(shown) < 12:            # show what differs, in plain numbers
                    for idx in torch.nonzero(got != exp)[:3].tolist():
                        ops = [float(sign * cell[c][tuple(idx)]) for c, sign in terms]
                        e, b_ = float(exp[tuple(idx)]), float(got[tuple(idx)])
                        ulp = float(torch.tensor(abs(e) if e else 1e-30).to(FP).nextafter(torch.tensor(float("inf"), dtype=FP)) - torch.tensor(abs(e)).to(FP)) if e else 2 ** -24
                        shown.append(f"        cells {ops} -> host {e:.9g}, board {b_:.9g}, off by {abs(e - b_):.3g} = {abs(e - b_) / ulp:.2f} fp16 steps of the result")
                differ += int((got != exp).sum())
            if differ:
                print(f"      level {L}: {differ} of {len(recipes) * n * k} sum values differ from the host; some of them:")
                print("\n".join(shown), flush=True)
            same /= len(recipes) * n * k
            results[L] = (t_st, t_on, t_prep, same, worst)
            check(f"{name}: level {L} sums made on chip match the host", worst <= 1.0,
                  f"{100 * same:.2f}% identical, worst {worst:.2f} fp16 steps")
            for label, C, online in ((f"stored level {L}", C_st, False), (f"online level {L}", C_on, True)):
                check(f"{name}: {label} finite", bool(torch.isfinite(C).all()), "NaN / Inf in the output")
                if kind == "if4":
                    ok = snr(truth, C) >= s0 - 0.5 and snr(plain, C) >= need(L, online)
                    detail = f"{snr(truth, C):.1f} dB vs bf16 (plain {s0:.1f}), {snr(plain, C):.1f} dB vs plain (needs {need(L, online)})"
                else:
                    ok = snr(truth, C) >= need(L, online)
                    detail = f"{snr(truth, C):.1f} dB vs the exact product (plain {s0:.1f}, needs {need(L, online)})"
                check(f"{name}: {label} accurate", ok, detail)

        ref = "vs bf16" if kind == "if4" else "vs exact"
        print(f"  {'':26s} {'board time':>12} {'vs plain':>9} {ref:>10} {'vs plain':>10} {'weights':>8}")
        for label, us, C, wb, prep in rows:
            print(f"  {label:26s} {us / 1e3:9.2f} ms {us / t0:8.2f}x {snr(truth, C):7.1f} dB"
                  + (f" {snr(plain, C):7.1f} dB" if C is not plain else " " * 11) + f" {wb:7.2f}x"
                  + (f"   (sums alone {prep[0] / 1e3:.2f} ms, {prep[1]} pieces)" if prep else ""), flush=True)
        for L in Ls:
            print(f"      level {L} sums made on chip: {100 * results[L][3]:.2f}% identical to the host, worst {results[L][4]:.2f} fp16 steps")
        summary.append((name, t0, results))

    verdict = lambda t, t0: "FASTER" if t < 0.995 * t0 else "slower" if t > 1.005 * t0 else "same"
    print(f"\nsummary, {NE} engine{'s' if NE > 1 else ''} (time relative to the plain matmul; below 1.00x is faster; online = 1.00x weight bytes)")
    print(f"  {'case':36s} {'stored L1':>10} {'online L1':>10} {'stored L2':>10} {'online L2':>10}   best online")
    for name, t0, res in summary:
        cell = lambda L, i: f"{res[L][i] / t0:9.2f}x" if L in res else f"{'-':>10}"
        best = min(res, key=lambda L: res[L][1])
        print(f"  {name:36s} {cell(1, 0)} {cell(1, 1)} {cell(2, 0)} {cell(2, 1)}   level {best}: {res[best][1] / t0:.2f}x {verdict(res[best][1], t0)}")
    bad = [c for c in checks if not c[1]]
    print(f"\nchecks: {len(checks) - len(bad)} of {len(checks)} passed")
    for what, _, detail in bad:
        print(f"  failed: {what}: {detail}")
    print("RESULT: " + ("PASS" if checks and not bad else "FAIL"))
    return 0 if checks and not bad else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL (the probe stopped with the error above)")
        code = 1
    sys.exit(code)
