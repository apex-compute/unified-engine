#!/usr/bin/env python3
"""Can decode read a 4-bit weight stored as input-column slabs, and what does it cost?

Strassen with online leaves needs the weight cut into cells. If the weight is stored as g slabs
W[:, slab j] (each [N, K / g], contiguous), every Strassen cell is a row range of a slab, so the
prefill needs no second copy. Decode (one row) then runs g matmuls per op, one per slab, each
adding onto the one before, instead of one matmul over the whole weight:

  whole      one quantized_matmat_core per engine on its N / engines output columns   (today)
  slabs g    g of them on K / g inputs each; slab j > 0 takes the running sum as a full-matrix
             bias (two output buffers, ping-pong); a fused activation goes on the last one

for the prefill / decode weight shapes, each op repeated --rep times in one program (28 = one
token's worth of that op). Per op: the time of each, the SNR of the slab result against the whole
one and of both against the exact product with the same 4-bit weights. The summary adds the ops
up to the extra time per decode token. RESULT: PASS means every slab result is finite and matches
the whole-weight result (>= 35 dB); the time cost is the summary.

  python models/alpamayo2_edge/utility/decode_slab_probe.py
  python models/alpamayo2_edge/utility/decode_slab_probe.py --slabs 2 4 --engines 8
"""
import argparse
import sys
import traceback

import torch

from strassen_probe_util import ACT_KW, BF, decode_if4, open_board, put as put_on, snr

# (name, K, N, fused activation, how many of it one decoder layer has)
OPS = [("q / o", 2048, 2048, {}, 2), ("k / v", 2048, 1024, {}, 2),
       ("fc1 half (ReLU)", 2048, 4608, ACT_KW["relu"], 2), ("fc2 half", 4608, 2048, {}, 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="max", help="engine count, or 'max' = every core the bitstream reports")
    ap.add_argument("--slabs", type=int, nargs="+", default=[2, 4], help="slab counts to try (4 = Strassen level 2, 2 = level 1)")
    ap.add_argument("--rep", type=int, default=28, help="times each op runs in its program (28 layers = one token)")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T, ue, NE, cores = open_board(args.dev, args.engines)
    from user_dma_core import TYPE
    base = ue.get_tensor_dram_addr()
    print(f"engines: {NE} of {cores}; one row, each op x {args.rep} per program")
    put = lambda t: put_on(ue, t)

    count = [0]

    def timed(label, body):
        """One program: body(E) on every engine, --rep times -> board time, us."""
        count[0] += 1
        print(f"  running: {label} ...", flush=True)               # if the board hangs, this names the culprit

        def fenced(E):
            E.ue.reset_isa_reg_counter()
            for _ in range(args.rep):
                body(E)
        ue._compile(f"slab{count[0]}", lambda: ue._round(fenced), parallel=True)
        return ue._exec(f"slab{count[0]}")

    def dot(E, k, cn, a, w, out, bias=None, act=None):
        E.ue.quantized_matmat_core(M=1, K=k, N=cn, A_DRAM_ADDR=a, B_DRAM_ADDR=w[0], OUTPUT_DRAM_ADDR=out,
                                   SCALE_DRAM_ADDR=w[1], data_type=TYPE.IF4,
                                   **(dict(C_DRAM_ADDR=bias, bias_mode="full_matrix") if bias is not None else {}), **(act or {}))

    t_empty = timed("empty round", lambda E: None)
    rows, bad, per_token = [], [], {g: 0.0 for g in args.slabs}
    whole_token = 0.0
    for name, K, N, act, per_layer in OPS:
        ue._tensor_dram_addr = base
        g_ = torch.Generator().manual_seed(args.seed)
        X = torch.randn(1, K, generator=g_).to(BF)
        W = torch.randn(N, K, generator=g_) / K ** 0.5
        pack = T.if4_pack(W)
        fn = (lambda v: v.clamp(min=0)) if act else (lambda v: v)
        exact = fn(X.double() @ decode_if4(pack[0], pack[1], N, K).T)
        a_x, a_w = put(X), ue._qstore(pack)
        a_o, a_p = ue._alloc(N, zero=True), ue._alloc(N, zero=True)        # result, and the ping-pong partner
        shard = ue._split(N)

        def whole(E):
            c0, cn = shard[E.e]
            if cn:
                dot(E, K, cn, a_x, ue._wshard(a_w, K, c0), a_o + c0 * 2, act=act)
        t0 = timed(f"{name}: whole weight", whole) - t_empty
        C0 = ue.dma_from_accelerator_memory(a_o, (1, N)).double()
        whole_token += t0 * per_layer / args.rep * 28
        rows.append((name, "whole", t0, None, snr(exact, C0)))

        for g in args.slabs:
            if K % (64 * g):
                continue
            k = K // g
            slabs = [ue._qstore(T.if4_pack(W[:, j * k:(j + 1) * k])) for j in range(g)]
            # slab j writes buffer j % 2 ... the last one must land in a_o
            outs = [a_o if (g - 1 - j) % 2 == 0 else a_p for j in range(g)]

            def chained(E, k=k, g=g, slabs=slabs, outs=outs):
                c0, cn = shard[E.e]
                for j in range(g if cn else 0):
                    dot(E, k, cn, a_x + j * k * 2, ue._wshard(slabs[j], k, c0), outs[j] + c0 * 2,
                        bias=outs[j - 1] + c0 * 2 if j else None, act=act if j == g - 1 else None)
            ue._zero(a_o, N), ue._zero(a_p, N)
            t = timed(f"{name}: {g} slabs", chained) - t_empty
            C = ue.dma_from_accelerator_memory(a_o, (1, N)).double()
            ok = bool(torch.isfinite(C).all()) and snr(C0, C) >= 35
            if not ok:
                bad.append(f"{name}, {g} slabs ({snr(C0, C):.1f} dB vs whole)")
            per_token[g] += t * per_layer / args.rep * 28
            rows.append((name, f"{g} slabs", t, snr(C0, C), snr(exact, C), t / t0))

    print(f"\n  {'op':18s} {'weight':9s} {'ms x ' + str(args.rep):>10} {'vs whole':>9} {'':8s} {'SNR vs whole':>13} {'SNR vs exact':>13}")
    for name, kind, t, s_whole, s_exact, *ratio in rows:
        r = ratio[0] if ratio else 1.0
        verdict = "" if not ratio else "faster" if r < 0.99 else "same" if r <= 1.01 else "slower"
        print(f"  {name:18s} {kind:9s} {t / 1e3:10.2f} {r:8.2f}x {verdict:8s} "
              + (f"{s_whole:10.1f} dB" if s_whole is not None else " " * 13) + f" {s_exact:10.1f} dB")
    print(f"\n  one decode token, these matmuls only (2 of each per layer, 28 layers): whole {whole_token / 1e3:.2f} ms")
    for g in args.slabs:
        if per_token[g]:
            d = per_token[g] - whole_token
            print(f"    {g} slabs: {per_token[g] / 1e3:.2f} ms  ({'+' if d >= 0 else ''}{d / 1e3:.2f} ms per token, {per_token[g] / whole_token:.2f}x)")
    print("\nRESULT: " + ("PASS" if not bad else "FAIL (slab result does not match the whole-weight result: " + "; ".join(bad) + ")"))
    return 0 if not bad else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL (the probe stopped with the error above)")
        code = 1
    sys.exit(code)
