#!/usr/bin/env python3
"""Every matmul shape of the vision tower, the projector and the prefill on the board, one engine,
as the native matmul and as Strassen level 1, 2 and 3 with fp16 intermediates.

  native              one matmat_mul_core: bf16 weights (vision, projector) or 4-bit IF4 (prefill)
  native, K in parts  the same work as narrower matmuls + adds (how the vision MLP runs today)
  Strassen level L    X, the weights' source and C stay as they are; the input sums S, the
                      products S . T^T and the recombination are stored in fp16. The weight sums T
                      are formed on the host in fp32 and stored
                        fp16 leaves   in fp16 (16-bit weights)
                        4-bit leaves  re-quantized to IF4, each leaf on its own (prefill only)

A width that does not halve L times into whole 64-lane blocks is ZERO-PADDED to the next multiple
of 64 * 2^L (the FLOPs column shows what that costs), so every level runs on every shape.
SNR is against the exact product: of the bf16 operands (vision, projector), of the bf16 input and
the unquantized weights (prefill). "time" is board time over native: below 1.00x is faster.

Operands sit in a recursive block layout (a matrix = its 4 quarters one after another, each laid
out the same way), so every Strassen sum is one flat elementwise op. The row gathers / strided
write-back the nn_lib kernel does around a level are NOT in these times, nor is a fused activation.

  python models/alpamayo2_edge/utility/strassen_fp16_probe.py                       # every shape
  python models/alpamayo2_edge/utility/strassen_fp16_probe.py --only prefill
  python models/alpamayo2_edge/utility/strassen_fp16_probe.py --shape 720 4352 1152 --weights bf16
"""
import argparse
import json
import os

import torch

from strassen_probe_util import (ACT_FN, ACT_KW, BF, FP, HERE, T_DEF, block_strassen, combo, from_blocks, hangs, leaves,
                                 open_board, products, put as put_on, quarters, run_regs, snr, to_blocks)


def shapes():
    """(name, rows on one engine, K, N, native weight type) of every matmul shape."""
    cfg = json.load(open(os.path.join(HERE, "alpamayo2_edge_config.json")))
    pj = cfg["connector"]
    return [("vision patch embed", 720, 768, 1152, "bf16"),
            ("vision q / k / o", 720, 1152, 1152, "bf16"),
            ("vision V^T (weight rows x keys)", 1152, 1152, 768, "bf16"),
            ("vision fc1", 720, 1152, 4352, "bf16"),
            ("vision fc2", 720, 4352, 1152, "bf16"),
            ("projector fc1", 360, pj["input_size"], pj["hidden_size"], "bf16"),
            ("projector fc2", 360, pj["hidden_size"], pj["output_size"], "bf16"),
            ("prefill q / o", 408, 2048, 2048, "if4"),
            ("prefill k / v", 408, 2048, 1024, "if4"),
            ("prefill fc1 (one half)", 408, 2048, 4608, "if4"),
            ("prefill fc2 (one half)", 408, 4608, 2048, "if4")]


def pad_to(t, rows, cols):
    out = torch.zeros(rows, cols, dtype=t.dtype)
    out[: t.shape[0], : t.shape[1]] = t
    return out


def host_model(X, W, L, leaf_w):
    """Strassen level L with the sums / products / recombination rounded to fp16 and the leaf
    weights (exact sums of W) stored by leaf_w; the top result is rounded to bf16."""
    return products(X, iter([leaf_w(w) for w, _ in leaves(W, L)]), L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run the shapes whose name contains this (vision, projector, prefill, fc2, ...)")
    ap.add_argument("--shape", type=int, nargs=3, default=None, metavar=("M", "K", "N"))
    ap.add_argument("--weights", choices=["bf16", "if4"], default="bf16", help="native weight type for --shape")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--kernel", action="store_true",
                    help="check the real kernel instead (nn_lib.strassen_matmul_core_dram, with a bias): SNR and time "
                         "against the native matmul at every level a shape allows without padding")
    ap.add_argument("--act", choices=["gelu", "silu", "relu"], default=None,
                    help="with --kernel (bf16): fuse this activation, in the native matmul and in the Strassen kernel "
                         "(activation= / identity_addr=); ends with RESULT: PASS / FAIL")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    todo = [("--shape", *args.shape, args.weights)] if args.shape else [s for s in shapes() if not args.only or args.only in s[0]]

    T, ue, _, _ = open_board(args.dev)
    import quant_lib
    from storage_format import StorageFormat, TensorFormats
    from user_dma_core import TYPE, UE_MODE
    base = ue.get_tensor_dram_addr()
    F16, B16 = StorageFormat.FP16, StorageFormat.BF16
    deq = lambda W: quant_lib.dequant("if4", *T.if4_pack(W.float())).double()        # what a 4-bit weight really is
    put = lambda t: put_on(ue, t)

    def run(emit):
        us, instr[0] = run_regs(ue, emit)
        return us

    instr = [0]                                                             # instructions of the last program run
    prog = lambda: f"  {instr[0]} instr ({instr[0] * 32} B)"

    def matmul(reg, m, k, n, a, w, out, fm):
        """w: a bf16 / fp16 weight address, or an IF4 (data, scales) pair."""
        kw = dict(is_B_quantized=True, data_type=TYPE.IF4, SCALE_DRAM_ADDR=w[1]) if isinstance(w, tuple) else {}
        ue.matmat_mul_core(M=m, K=k, N=n, A_DRAM_ADDR=a, B_DRAM_ADDR=w[0] if isinstance(w, tuple) else w,
                           OUTPUT_DRAM_ADDR=out, gpr_M_reg=reg(m), formats=fm, **kw)

    if args.kernel and args.weights == "if4":
        # The real kernel on a 4-bit layer: nn_lib.strassen_weight_tree_if4 (plain blocks IF4, sums IF8)
        # against the plain bf16 and the plain IF4 matmul; "sums IF4" re-quantizes every leaf to 4 bits.
        import nn_lib
        cases = [("--shape", *args.shape)] if args.shape else [
            ("prefill q / o", 408, 2048, 2048), ("prefill k / v", 408, 2048, 1024),
            ("prefill fc1 (one half)", 408, 2048, 4608), ("prefill fc2 (one half)", 408, 4608, 2048)]
        print("nn_lib.strassen_matmul_core_dram on a 4-bit layer (fp16 intermediates), one engine")
        for name, M, K, N in cases:
            ue._tensor_dram_addr = base
            g = torch.Generator().manual_seed(args.seed)
            X = torch.randn(M, K, generator=g).to(BF)
            W = torch.randn(N, K, generator=g) / K ** 0.5
            pack = T.if4_pack(W)
            a_x, a_wb, a_wq = put(X), put(W.to(BF)), ue._qstore(pack)
            a_c = ue.allocate_tensor_dram(M * N * 2, align_bytes=128)
            t_bf = run(lambda reg: matmul(reg, M, K, N, a_x, a_wb, a_c, None))
            ref_bf = ue.dma_from_accelerator_memory(a_c, (M, N)).double()
            t_q = run(lambda reg: matmul(reg, M, K, N, a_x, a_wq, a_c, None))
            ref_q = ue.dma_from_accelerator_memory(a_c, (M, N)).double()
            size = len(pack[0]) + len(pack[1])
            print(f"\n{name}: X [{M}, {K}] . W [{N}, {K}]^T")
            print(f"  {'':24s} {'board ms':>9} {'vs IF4':>7} {'SNR vs bf16':>12} {'SNR vs IF4':>11} {'weights':>8}")
            print(f"  {'plain bf16':24s} {t_bf / 1e3:9.2f} {t_bf / t_q:6.2f}x {'':>12} {'':>11} {N * K * 2 / size:7.2f}x")
            print(f"  {'plain IF4 (today)':24s} {t_q / 1e3:9.2f} {1.0:6.2f}x {snr(ref_bf, ref_q):9.1f} dB {'':>11} {1.0:7.2f}x", flush=True)
            mark = ue.get_tensor_dram_addr()
            for L in range(1, nn_lib.strassen_levels(K, N, args.levels) + 1):
                if M % (1 << L):
                    continue
                for prec in ("if8", "if4"):
                    ue._tensor_dram_addr = mark
                    nbytes = [0]

                    def store_pack(d, s_, n, k, t):
                        nbytes[0] += len(d) + len(s_)
                        return ue._qstore((d, s_, n, k))
                    tree = nn_lib.strassen_weight_tree_if4(pack[0], pack[1], N, K, None, L, store_pack, None, prec)
                    scratch = [(ue._alloc(x, zero=True), ue._alloc(p_, zero=True)) for x, p_ in nn_lib.strassen_scratch_elements(M, K, N, L)]
                    a_o = ue.allocate_tensor_dram(M * N * 2, align_bytes=128)
                    us = run(lambda reg: nn_lib.strassen_matmul_core_dram(ue, M, K, N, a_x, tree, a_o, scratch))
                    C = ue.dma_from_accelerator_memory(a_o, (M, N)).double()
                    label = f"level {L}, sums {'IF8' if prec == 'if8' else 'IF4'}"
                    print(f"  {label:24s} {us / 1e3:9.2f} {us / t_q:6.2f}x {snr(ref_bf, C):9.1f} dB {snr(ref_q, C):8.1f} dB {nbytes[0] / size:7.2f}x"
                          + prog() + ("" if bool(torch.isfinite(C).all()) else "  NaN/Inf!"), flush=True)
        return

    if args.kernel:
        import nn_lib
        cases = [("--shape", *args.shape)] if args.shape else [
            ("vision o", 720, 1152, 1152), ("vision fc2", 720, 4352, 1152), ("2048 -> 2048", 408, 2048, 2048), ("4608 -> 2048", 408, 4608, 2048)]
        print("nn_lib.strassen_matmul_core_dram (fp16 intermediates, fp16 leaves, bias folded in), one engine")
        act, fn = ACT_KW[args.act], ACT_FN[args.act]
        bad = []
        for name, M, K, N in cases:
            ue._tensor_dram_addr = base
            g = torch.Generator().manual_seed(args.seed)
            X = torch.randn(M, K, generator=g).to(BF)
            W = (torch.randn(N, K, generator=g) / K ** 0.5).to(BF).float()
            b = torch.randn(N, generator=g).to(BF).float()
            exact = fn(X.double() @ W.double().T + b.double())
            a_x, a_w, a_b = put(X), put(W.to(BF)), put(b.to(BF))
            a_eye = put(torch.eye(64, dtype=BF)) if act else None
            a_c = ue.allocate_tensor_dram(M * N * 2, align_bytes=128)
            t0 = run(lambda reg: ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a_x, B_DRAM_ADDR=a_w, OUTPUT_DRAM_ADDR=a_c,
                                                    C_DRAM_ADDR=a_b, bias_mode="broadcast_N", gpr_M_reg=reg(M), **act))
            native, n_prog = ue.dma_from_accelerator_memory(a_c, (M, N)).double(), prog()
            print(f"\n{name}: {args.act or ''}(X [{M}, {K}] . W [{N}, {K}]^T + b)")
            print(f"  {'':12s} {'board ms':>9} {'time':>6} {'vs exact':>9} {'vs native':>10}")
            print(f"  {'native':12s} {t0 / 1e3:9.2f} {1.0:5.2f}x {snr(exact, native):6.1f} dB {'':>10}" + n_prog, flush=True)
            mark = ue.get_tensor_dram_addr()
            for L in range(1, nn_lib.strassen_levels(K, N, args.levels) + 1):
                if M % (1 << L):
                    continue
                ue._tensor_dram_addr = mark
                tree = nn_lib.strassen_weight_tree(W, b, L, put, put)
                scratch = [(ue._alloc(x, zero=True), ue._alloc(p_, zero=True)) for x, p_ in nn_lib.strassen_scratch_elements(M, K, N, L)]
                a_o = ue.allocate_tensor_dram(M * N * 2, align_bytes=128)
                us = run(lambda reg: nn_lib.strassen_matmul_core_dram(ue, M, K, N, a_x, tree, a_o, scratch,
                                                                      activation=act or None, identity_addr=a_eye))
                C = ue.dma_from_accelerator_memory(a_o, (M, N)).double()
                if not bool(torch.isfinite(C).all()) or snr(exact, C) < snr(exact, native) - 6:
                    bad.append(f"{name} level {L}")
                print(f"  {'level ' + str(L):12s} {us / 1e3:9.2f} {us / t0:5.2f}x {snr(exact, C):6.1f} dB {snr(native, C):7.1f} dB"
                      + prog() + ("" if bool(torch.isfinite(C).all()) else "  NaN/Inf!"), flush=True)
        print("\nRESULT: " + ("PASS" if not bad else "FAIL (more than 6 dB below the native matmul, or not finite: " + ", ".join(bad) + ")"))
        return

    summary = []
    for name, M, K, N, wtype in todo:
        ue._tensor_dram_addr = base                                         # every shape reuses the tensor arena
        g = torch.Generator().manual_seed(args.seed)
        X = torch.randn(M, K, generator=g).to(BF)
        W = torch.randn(N, K, generator=g) / K ** 0.5
        W = W.to(BF).float() if wtype == "bf16" else W                      # prefill: the unquantized weight is the truth
        exact = X.double() @ W.double().T
        print(f"\n{name}: X [{M}, {K}] . W [{N}, {K}]^T, native weights {wtype}, one engine", flush=True)
        print(f"  {'':46s} {'FLOPs':>6} {'board ms':>9} {'time':>6} {'SNR':>8} {'predicted':>10}")
        done = []

        def line(label, ratio, pred, C, us):
            t0 = done[0][1] if done else us
            done.append((label, us))
            print(f"  {label:46s} {ratio:5.2f}x {us / 1e3:9.2f} {us / t0:5.2f}x {snr(exact, C):5.1f} dB {snr(exact, pred):7.1f} dB"
                  + ("" if bool(torch.isfinite(C).all()) else "  NaN/Inf!"), flush=True)

        # ---- native
        a_x, a_c = put(X), ue._alloc(M * N)
        a_w = put(W.to(BF)) if wtype == "bf16" else ue._qstore(T.if4_pack(W))
        assert wtype == "if4" or not hangs(M, K, N)
        us = run(lambda reg: matmul(reg, M, K, N, a_x, a_w, a_c, None))
        Wn = W.double() if wtype == "bf16" else deq(W)
        line(f"native ({wtype})", 1.0, (X.double() @ Wn.T).to(BF), ue.dma_from_accelerator_memory(a_c, (M, N)).double(), us)

        # ---- native with K in parts (bf16): the wide-input split
        parts = next((p for p in (4, 2) if K % (64 * p) == 0 and K // p >= 1024), 1) if wtype == "bf16" else 1
        if parts > 1:
            kp, pred = K // parts, None
            for j in range(parts):
                p_j = (X[:, j * kp:(j + 1) * kp].double() @ W[:, j * kp:(j + 1) * kp].double().T).to(BF).double()
                pred = p_j if pred is None else (pred + p_j).to(BF).double()
            a_xp = [put(X[:, j * kp:(j + 1) * kp]) for j in range(parts)]
            a_wp = [put(W[:, j * kp:(j + 1) * kp].to(BF)) for j in range(parts)]
            a_p, a_acc = ue._alloc(M * N), [ue._alloc(M * N) for _ in range(2)]

            def emit(reg):
                acc = None
                for j in range(parts):
                    matmul(reg, M, kp, N, a_xp[j], a_wp[j], a_acc[0] if j == 0 else a_p, None)
                    if j:
                        ue.eltwise_core_dram(M=M, N=N, dram_a=acc, dram_b=a_p, dram_out=a_acc[j % 2], mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=reg(M))
                    acc = a_acc[j % 2]
                emit.out = acc
            us = run(emit)
            line(f"native, {parts} x K {kp} + adds", 1.0, pred, ue.dma_from_accelerator_memory(emit.out, (M, N)).double(), us)

        # ---- Strassen levels with fp16 intermediates, zero-padded where a width does not halve
        mark = ue.get_tensor_dram_addr()                                    # every variant reuses the arena above the natives
        for L in range(1, args.levels + 1):
            unit = 64 << L
            Kp, Np = -(-K // unit) * unit, -(-N // unit) * unit
            if M % (1 << L):
                print(f"  Strassen level {L}: {M} rows do not halve {L} times, skipped")
                continue
            ratio = (7 / 8) ** L * (Kp * Np) / (K * N)
            Xp, Wp = pad_to(X, M, Kp), pad_to(W.float(), Np, Kp)
            pad = "" if (Kp, Np) == (K, N) else f", padded {Kp} -> {Np}"
            leaf = (M >> L, Kp >> L, Np >> L)
            for kind in (("fp16", "if4") if wtype == "if4" else ("fp16",)):
                label = f"Strassen L{L}, {'4-bit' if kind == 'if4' else 'fp16'} leaves{pad}"
                if kind == "fp16" and hangs(*leaf):
                    print(f"  {label}: leaf matmul {leaf} is the shape that hangs the board, skipped")
                    continue
                pred = host_model(Xp.double(), Wp.double(), L, (lambda t: t.to(FP).double()) if kind == "fp16" else deq)[:, :N]
                store = (lambda t: put(t.to(FP))) if kind == "fp16" else (lambda t: ue._qstore(T.if4_pack(t)))

                def tree(Wn, lv):                                           # weight sums in fp32 on the host, one store per leaf
                    if lv == 0:
                        return store(Wn)
                    wq = quarters(Wn)
                    return [tree(combo(wq, *t), lv - 1) for t in T_DEF]

                ue._tensor_dram_addr = mark
                a_xz, a_t, a_out = put(to_blocks(Xp, L)), tree(Wp, L), ue._alloc(M * Np)

                def leaf_mm(reg, m, k, n, x, t, out, xf, of):
                    matmul(reg, m, k, n, x, t, out, TensorFormats(a=xf, b=B16 if isinstance(t, tuple) else F16, output=of))

                us = run(lambda reg: block_strassen(ue, reg, leaf_mm, a_xz, B16, a_t, a_out, B16, M, Kp, Np, L))
                C = from_blocks(ue.dma_from_accelerator_memory(a_out, (M * Np,)).double(), M, Np, L)[:, :N]
                line(label, ratio, pred, C, us)

        t0, best = done[0][1], min(done, key=lambda d: d[1])
        print(f"  -> fastest: {best[0]} ({best[1] / t0:.2f}x of native)", flush=True)
        summary.append((name, done))

    print("\nsummary (board time over native; below 1.00x is faster)")
    for name, done in summary:
        t0, best = done[0][1], min(done, key=lambda d: d[1])
        print(f"  {name:34s} " + "  ".join(f"{lab.replace('Strassen ', '').replace(' leaves', '').split(', padded')[0]} {us / t0:.2f}x"
                                          + ("*" if (lab, us) == best else "") for lab, us in done[1:]))


if __name__ == "__main__":
    main()
