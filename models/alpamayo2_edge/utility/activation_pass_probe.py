#!/usr/bin/env python3
"""Can the board apply an activation to a buffer WITHOUT a matmul in front of it?

Strassen builds its output in four quarters. Two end in a product, so the matmul's own fused
activation covers them. The other two are only brought to the output format by one elementwise
pass (x * 1.0, fp16 in, bf16 out). If that pass can carry the activation, a matmul with a fused
GELU / SiLU / ReLU can use Strassen at no extra cost. This probe asks the hardware, one engine,
on one buffer of known values:

  baseline         the pass as the kernel emits it today (x * 1.0, no activation): must return x
  A  broadcast     the same pass with the activation fields set       <- the free route, if honoured
  B  add zero      an elementwise add of a zero buffer with the activation fields set
  C  identity dot  an on-chip identity matrix-vector product per 64 values with the activation
                   fields set (the matmul epilogue, without the DRAM round trip)  <- slower fallback
  D  looped C      nn_lib._strassen_activate in place on a DRAM buffer: route C with the products
                   and the pieces in hardware loops, against route C unrolled over the same buffer
                   (one product per 64 values), on one piece and on a multi-piece buffer

for GELU (x * sigmoid(1.702 x), the hardware's), SiLU and ReLU. Per method and activation it says
WORKS (the result is the activation), IGNORED (the result is x: the field does nothing there) or
WRONG, with the SNR against the host and the time per value. RESULT: PASS means the probe itself
ran cleanly, the baseline is exact and every route D program returns the activation (30 dB or
more, as good as the unrolled form); which of the routes A-C work is the table above it. Route D
prints the size of every program (instructions, bytes at 32 B per instruction).

  python models/alpamayo2_edge/utility/activation_pass_probe.py
  python models/alpamayo2_edge/utility/activation_pass_probe.py --looped-only     # baseline and route D
"""
import argparse
import sys
import traceback

import torch

from strassen_probe_util import ACT_FN, ACT_KW, BF, FP, open_board, put as put_on, snr

ACTS = {a: f for a, f in ACT_FN.items() if a}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elements", type=int, default=65536, help="values in the buffer (a multiple of 64, at most 65536)")
    ap.add_argument("--looped-only", action="store_true", help="skip routes A-C")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    E = args.elements
    assert E % 64 == 0 and 64 <= E <= 65536
    rows = E // 64

    _, ue, _, _ = open_board(args.dev)
    import nn_lib
    import user_dma_core
    from storage_format import StorageFormat, TensorFormats
    from user_dma_core import BROADCAST_MODE, UE_MODE, URAM_SECTION, URAM_WRITE_SRC
    F16, B16 = StorageFormat.FP16, StorageFormat.BF16
    fm = TensorFormats(a=F16, b=B16, output=B16)              # fp16 in (zeros / identity are bf16), bf16 out
    A, WB = URAM_SECTION.URAM_A.value, URAM_WRITE_SRC.URAM_WRITE_BACK.value
    OUT_SRAM = 0x40000                                         # where route C writes inside URAM A, clear of the input
    put = lambda t: put_on(ue, t)

    x = (torch.randn(E, generator=torch.Generator().manual_seed(args.seed)) * 2.0).to(FP)
    a_x, a_zero, a_eye = put(x), put(torch.zeros(E, dtype=BF)), put(torch.eye(64, dtype=BF))
    a_out = ue._alloc(E)
    xd = x.double()

    def lalu(act):                                             # exactly what the matmul sets for its fused epilogue
        if act is None:
            return user_dma_core.LALU_MODE.BYPASS, 0, 0
        return ue._matmul_activation(act == "gelu", act == "silu", False, act == "relu", False, 0.0, float("inf"), formats=fm)

    def broadcast(act):                                        # route A / baseline: x * 1.0, as start_queue_broadcast emits it
        mode, la, lb = lalu(act)
        ue.ue_arithmetic_op(BROADCAST_MODE.SCALAR_IN_REG.value, 0, 1, la, lb, mode.value,
                            ue.float_to_storage(1.0, storage_format=fm.scalar), A, 0, 0, WB,
                            UE_MODE.MUL_BROADCAST, 0, 0, 0, rows, 0, 0, 0, formats=fm)

    def add_zero(act):                                         # route B: x + 0, as start_queue_eltwise emits it
        mode, la, lb = lalu(act)
        ue.ue_arithmetic_op(0, 0, 1, la, lb, mode.value, 0, A, 0, 0, WB, UE_MODE.ELTWISE_ADD, 0, 0, 0, rows, 0, 0, 0, formats=fm)

    def identity_dot(act, n_rows):                             # route C: one identity matrix-vector product per 64 values
        mode, la, lb = lalu(act)
        for r_ in range(n_rows):
            ue.start_queue_for_bf16_matvec_operation(0, 0, r_ * 128, 0x80000, OUT_SRAM + r_ * 128, 64, 64,
                                                     lalu_mode=mode, lalu_a=la, lalu_b=lb, formats=fm)

    def run(label, emit, n=E, from_sram=0x00000, loads=("x",)):
        """Load, run emit, store; -> (board us, the first n values as the board left them)."""
        print(f"  running: {label} ...", flush=True)                # if the board hangs, this names the culprit

        def body():
            ue.accelerator_memory_to_sram(accelerator_dram_address=a_x, sram_address=0x00000, element_size=E)
            if "zero" in loads:
                ue.accelerator_memory_to_sram(accelerator_dram_address=a_zero, sram_address=0x80000, element_size=E)
            if "eye" in loads:
                ue.accelerator_memory_to_sram(accelerator_dram_address=a_eye, sram_address=0x80000, element_size=64 * 64)
            emit()
            ue.sram_to_accelerator_memory(from_sram, a_out, n)
        us = ue._run_transient(body, 60.0)
        return us, ue.dma_from_accelerator_memory(a_out, (n,)).double()

    print(f"activation on a buffer of {E} fp16 values, written back as bf16, one engine\n")
    t_copy, _ = run("load + store only", lambda: None)
    t_base, y = run("baseline, x * 1.0", lambda: broadcast(None))
    # x rounded to bf16: identical, or within one bf16 step where the board breaks a tie the other way
    exact = float((y == x.to(BF).double()).double().mean())
    off = float(((y - xd).abs() / xd.abs().clamp_min(1e-30)).max())
    base_ok = bool(torch.isfinite(y).all()) and off <= 2 ** -7
    print(f"  baseline x * 1.0: {'returns x' if base_ok else 'DOES NOT return x'} ({100 * exact:.2f}% identical to the host's bf16 "
          f"rounding, worst relative difference {off:.2g}); load + store {t_copy / 1e3:.3f} ms, the pass itself {(t_base - t_copy) / 1e3:.3f} ms\n")

    NC = min(rows, 128)                                        # route C is one op per 64 values: time a part of the buffer
    table, crashed = [], None
    try:
        for act, fn in ({} if args.looped_only else ACTS).items():
            want = fn(xd)
            for route, label, emit, n, src, loads in (
                    ("A broadcast", f"A: x * 1.0 with {act}", lambda act=act: broadcast(act), E, 0x00000, ("x",)),
                    ("B add zero", f"B: x + 0 with {act}", lambda act=act: add_zero(act), E, 0x00000, ("x", "zero")),
                    ("C identity dot", f"C: identity dot with {act}", lambda act=act: identity_dot(act, NC), NC * 64, OUT_SRAM, ("x", "eye"))):
                crashed = label
                us, y = run(label, emit, n, src, loads)
                ref, plain = want[:n], x.to(BF).double()[:n]
                good, same = snr(ref, y), snr(plain, y) if bool(plain.abs().sum()) else 0.0
                finite = bool(torch.isfinite(y).all())
                verdict = "WORKS" if finite and good >= 30 else "IGNORED" if finite and same >= 60 else "WRONG"
                table.append((route, act, verdict, good, same, (us - t_copy) * 1e3 / n, float((ref - y).abs().max())))
        crashed = None
    except Exception:
        traceback.print_exc()

    print(f"\n  {'route':16s} {'activation':10s} {'verdict':8s} {'SNR vs activation':>18} {'SNR vs plain x':>15} {'ns per value':>13} {'worst error':>12}")
    for route, act, verdict, good, same, ns, worst in table:
        print(f"  {route:16s} {act:10s} {verdict:8s} {good:15.1f} dB {same:12.1f} dB {ns:13.2f} {worst:12.4g}")
    print("\n  what works:")
    for route in ("A broadcast", "B add zero", "C identity dot"):
        ok = [a for r_, a, v, *_ in table if r_ == route and v == "WORKS"]
        ns = [n_ for r_, a, v, _, _, n_, _ in table if r_ == route and v == "WORKS"]
        print(f"    {route:16s} " + (f"{', '.join(ok)} at about {sum(ns) / len(ns):.2f} ns per value" if ok else "nothing"))
    if crashed:
        print(f"\nRESULT: FAIL (stopped in '{crashed}'; if it was a timeout the board needs a reset)")
        return 1

    # Route D: the kernel's own pass, in place on DRAM, against route C unrolled over the same buffer.
    def unrolled(act, a_buf, n):                               # per piece: load, one identity product per 64 values, store
        P = nn_lib.STRASSEN_PIECE_ELEMENTS
        for e0 in range(0, n, P):
            take = min(P, n - e0)
            ue.accelerator_memory_to_sram(accelerator_dram_address=a_buf + e0 * 2, sram_address=0x00000, element_size=take)
            ue.accelerator_memory_to_sram(accelerator_dram_address=a_eye, sram_address=0x80000, element_size=64 * 64)
            identity_dot(act, take // 64)
            ue.sram_to_accelerator_memory(OUT_SRAM, a_buf + e0 * 2, take)
    FORMS = (("unrolled", unrolled),
             ("looped (the kernel)", lambda act, a_buf, n: nn_lib._strassen_activate(ue, a_buf, n, ACT_KW[act], a_eye, B16)))
    d_bad = []
    print(f"\n  route D: nn_lib._strassen_activate in place (time includes its loads and stores)")
    print(f"  {'values':>8} {'activation':10s} {'program':22s} {'instr':>7} {'bytes':>9} {'SNR vs activation':>18} {'SNR vs unrolled':>16} {'ns per value':>13}")
    try:
        for n in (E, 3 * 65536 + 4096):                         # one piece; three full pieces and a short last one
            xn = (torch.randn(n, generator=torch.Generator().manual_seed(args.seed + 1)) * 2.0).to(FP)
            a_buf = ue._alloc(n)
            for act, fn in ACTS.items():
                want, first = fn(xn.double()), None
                for name, emit in FORMS:
                    crashed = f"D: {name}, {act}, {n} values"
                    print(f"  running: {crashed} ...", flush=True)
                    ue.dma_to_accelerator_memory(a_buf, xn)
                    count = [0]

                    def body(act=act, a_buf=a_buf, n=n, count=count, emit=emit):
                        emit(act, a_buf, n)
                        count[0] = ue.capture_count
                    us = ue._run_transient(body, 60.0)
                    y = ue.dma_from_accelerator_memory(a_buf, (n,)).double()
                    reference = first is None
                    first = y if reference else first
                    good, same = snr(want, y), snr(first, y)
                    ok = bool(torch.isfinite(y).all()) and good >= 30 and (reference or same >= good - 1)
                    if not ok:
                        d_bad.append(crashed)
                    print(f"  {n:8d} {act:10s} {name:22s} {count[0]:7d} {count[0] * 32:9d} {good:15.1f} dB {same:13.1f} dB "
                          f"{us * 1e3 / n:13.2f}  {'ok' if ok else 'BAD'}", flush=True)
        crashed = None
    except Exception:
        traceback.print_exc()
    if crashed:
        print(f"\nRESULT: FAIL (stopped in '{crashed}'; if it was a timeout the board needs a reset)")
        return 1
    if d_bad:
        print(f"\nRESULT: FAIL (route D wrong in: {'; '.join(d_bad)})")
        return 1
    print("\nRESULT: " + ("PASS" if base_ok else "FAIL (the baseline pass does not return x)"))
    return 0 if base_ok else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        print("RESULT: FAIL (the probe stopped with the error above)")
        code = 1
    sys.exit(code)
