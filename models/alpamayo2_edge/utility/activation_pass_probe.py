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

for GELU (x * sigmoid(1.702 x), the hardware's), SiLU and ReLU. Per method and activation it says
WORKS (the result is the activation), IGNORED (the result is x: the field does nothing there) or
WRONG, with the SNR against the host and the time per value. RESULT: PASS means the probe itself
ran cleanly and the baseline is exact; which routes work is the table above it.

  python models/alpamayo2_edge/utility/activation_pass_probe.py
"""
import argparse
import os
import sys
import traceback

import torch

BF, FP = torch.bfloat16, torch.float16
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTS = {"gelu": lambda x: x * torch.sigmoid(1.702 * x), "silu": lambda x: x * torch.sigmoid(x), "relu": lambda x: x.clamp(min=0)}


def snr(ref, x):
    ref, x = ref.double(), x.double()
    return float(10 * torch.log10(ref.pow(2).sum() / (ref - x).pow(2).sum().clamp_min(1e-300)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elements", type=int, default=65536, help="values in the buffer (a multiple of 64, at most 65536)")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    E = args.elements
    assert E % 64 == 0 and 64 <= E <= 65536
    rows = E // 64

    sys.path.insert(0, HERE)
    import alpamayo2_edge_test as T
    import user_dma_core
    from storage_format import StorageFormat, TensorFormats
    from user_dma_core import BROADCAST_MODE, UE_MODE, URAM_SECTION, URAM_WRITE_SRC
    dev = args.dev or T.CFG["defaults"]["dev"]
    user_dma_core.set_dma_device(dev)
    user_dma_core.configure_clock_from_hardware()
    ue = T.Alpamayo2Edge_UnifiedEngine(dev=dev, engines=1)
    F16, B16 = StorageFormat.FP16, StorageFormat.BF16
    fm = TensorFormats(a=F16, b=B16, output=B16)              # fp16 in (zeros / identity are bf16), bf16 out
    A, WB = URAM_SECTION.URAM_A.value, URAM_WRITE_SRC.URAM_WRITE_BACK.value
    OUT_SRAM = 0x40000                                         # where route C writes inside URAM A, clear of the input

    def put(t):
        a = ue._alloc(t.numel())
        ue.dma_to_accelerator_memory(a, t.contiguous().flatten())
        return a

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
        for act, fn in ACTS.items():
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
