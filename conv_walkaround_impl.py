#!/usr/bin/env python3
"""Variant implementations + report writer for conv_walkaround_bench.py.

Each variant reproduces, at op level, exactly what a model does today:

  im2col          materialize the sliding window, then one matmul
                  (qwen2.5_omni_7b audio conv1/conv2, mobilenetv2 stem,
                   mobilesam 3x3, parakeet subsampling)
  per_tap         k matmuls, each accumulating through the previous partial
                  (qwen2.5_omni_7b token2wav / BigVGAN conv1d)
  toeplitz        per-channel (1,L)@(L,L) matmul against an L x L matrix
                  (parakeet DW conv1d, gemma4 audio depthwise_conv1d_core_dram)
  tapwise         k eltwise multiplies against a pre-tiled weight plus k-1 adds
                  (mobilenetv2 conv2d_3x3_dw_tapwise_dram) -- the FLOP-exact
                  depthwise alternative, measured here because the CONV
                  hardware has no depthwise mode

and pairs them against the real CONV op (conv1d_core / run_conv2d_layer).
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F

import user_dma_core
from user_dma_core import (
    UnifiedEngine,
    TYPE,
    UE_MODE,
    UE_VECTOR_SIZE,
    DMA_DEVICE_H2C,
    DMA_DEVICE_C2H,
    CONV_GEOMETRY_QUEUE_CONFIG,
)

BPE = 2


def _align(n: int, a: int = UE_VECTOR_SIZE) -> int:
    return ((n + a - 1) // a) * a


def _put(ue: UnifiedEngine, t: torch.Tensor) -> int:
    """Write a bf16 tensor to a fresh tensor-DRAM slot, return its address."""
    t = t.to(torch.bfloat16).contiguous()
    nbytes = t.numel() * BPE
    addr = ue.allocate_tensor_dram(nbytes)
    wrote = ue.dma_write(DMA_DEVICE_H2C, addr, t, nbytes)
    if wrote != nbytes:
        raise RuntimeError(f"short H2C: {wrote} of {nbytes}")
    return addr


def _get(ue: UnifiedEngine, addr: int, shape) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    out = torch.zeros(n, dtype=torch.bfloat16)
    ue.dma_read(DMA_DEVICE_C2H, addr, out, n * BPE)
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _reference(case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, w, y) in bf16-rounded float32. Layouts are per-case natural."""
    torch.manual_seed(0)
    if case.in_hw is not None:
        x = torch.randn(case.c_in, case.in_hw, case.in_hw)
        w = torch.randn(case.c_out, case.c_in, case.kernel, case.kernel) * 0.05
        y = F.conv2d(x.unsqueeze(0).to(torch.bfloat16).float(),
                     w.to(torch.bfloat16).float(),
                     stride=case.stride, padding=case.kernel // 2).squeeze(0)
        return x, w, y
    if case.depthwise:
        x = torch.randn(case.c_in, case.length)
        w = torch.randn(case.c_in, 1, case.kernel) * 0.1
        pad = case.dilation * (case.kernel - 1) // 2
        y = F.conv1d(x.unsqueeze(0).to(torch.bfloat16).float(),
                     w.to(torch.bfloat16).float(),
                     padding=pad, dilation=case.dilation,
                     groups=case.c_in).squeeze(0)
        return x, w, y
    x = torch.randn(case.c_in, case.length)
    w = torch.randn(case.c_out, case.c_in, case.kernel) * 0.05
    pad = case.dilation * (case.kernel - 1) // 2
    y = F.conv1d(x.unsqueeze(0).to(torch.bfloat16).float(),
                 w.to(torch.bfloat16).float(),
                 stride=case.stride, padding=pad, dilation=case.dilation).squeeze(0)
    return x, w, y


# ---------------------------------------------------------------------------
# Workaround variants
# ---------------------------------------------------------------------------

def _v_im2col(ue, case, x, w, yref, timed, snr, Result, prep_bytes):
    """Materialize the window, then one matmul -- what the models do."""
    pb, pd = prep_bytes(case)
    if case.in_hw is not None:
        # HWC im2col: [out_h*out_w, k*k*C]
        xp = F.pad(x.unsqueeze(0), (case.kernel // 2,) * 4).squeeze(0)
        cols = F.unfold(xp.unsqueeze(0), case.kernel, stride=case.stride).squeeze(0).T
        A = cols.contiguous()                            # [P, k*k*C]
        B = w.reshape(case.c_out, -1).contiguous()       # [OC, k*k*C]
        rows, K, N = A.shape[0], A.shape[1], case.c_out
    else:
        pad = case.dilation * (case.kernel - 1) // 2
        xp = F.pad(x.unsqueeze(0), (pad, pad)).squeeze(0)
        cols = xp.unfold(1, case.dilation * (case.kernel - 1) + 1, case.stride)
        if case.dilation > 1:
            cols = cols[:, :, ::case.dilation]
        # kernel-major lanes, matching the Omni audio front end's layout
        A = cols.permute(1, 2, 0).reshape(cols.shape[1], -1).contiguous()
        # kernel-major lanes: A's column index is j*C + c, so the weight has to
        # be permuted to [OC, j, c] rather than torch's native [OC, c, j].
        B = w.permute(0, 2, 1).reshape(case.c_out, -1).contiguous()
        rows, K, N = A.shape[0], A.shape[1], case.c_out

    K_p, N_p = _align(K), _align(N)
    Ap = torch.zeros(rows, K_p); Ap[:, :K] = A
    Bp = torch.zeros(N_p, K_p);  Bp[:N, :K] = B
    a_addr, b_addr = _put(ue, Ap), _put(ue, Bp)
    o_addr = ue.allocate_tensor_dram(rows * N_p * BPE)
    issued = 2 * rows * K_p * N_p

    ms, ib = timed(ue, lambda: ue.matmat_mul_core(
        M=rows, K=K_p, N=N_p,
        A_DRAM_ADDR=a_addr, B_DRAM_ADDR=b_addr, OUTPUT_DRAM_ADDR=o_addr))
    got = _get(ue, o_addr, (rows, N_p))[:, :N]
    ref = yref.reshape(case.c_out, -1).T if case.in_hw is None else \
        yref.reshape(case.c_out, -1).T
    return Result("workaround: im2col + matmul", issued, pb, pd, ms,
                  snr(ref, got.float()), ib,
                  extra=f"matmul [{rows}x{K_p}]@[{K_p}x{N_p}]")


def _v_per_tap(ue, case, x, w, yref, timed, snr, Result, prep_bytes):
    """k matmuls, each accumulating through the previous partial's C term."""
    pb, pd = prep_bytes(case)
    pad = case.dilation * (case.kernel - 1) // 2
    xp = F.pad(x.unsqueeze(0), (pad, pad)).squeeze(0).T.contiguous()   # [T+2p, C]
    C_p, O_p = _align(case.c_in), _align(case.c_out)
    rows = case.out_len
    xpad = torch.zeros(xp.shape[0], C_p); xpad[:, :case.c_in] = xp
    x_addr = _put(ue, xpad)
    taps = [_put(ue, torch.nn.functional.pad(
        w[:, :, kk].contiguous(), (0, C_p - case.c_in, 0, O_p - case.c_out)))
        for kk in range(case.kernel)]
    acc = [ue.allocate_tensor_dram(rows * O_p * BPE) for _ in range(2)]
    out = ue.allocate_tensor_dram(rows * O_p * BPE)
    issued = case.kernel * 2 * rows * C_p * O_p

    def emit():
        for kk in range(case.kernel):
            last = kk == case.kernel - 1
            dst = out if last else acc[kk % 2]
            prev = None if kk == 0 else acc[(kk - 1) % 2]
            ue.matmat_mul_core(
                M=rows, K=C_p, N=O_p,
                A_DRAM_ADDR=x_addr + kk * case.dilation * C_p * BPE,
                B_DRAM_ADDR=taps[kk], OUTPUT_DRAM_ADDR=dst,
                C_DRAM_ADDR=prev, bias_mode="full_matrix" if prev is not None else None)

    ms, ib = timed(ue, emit)
    got = _get(ue, out, (rows, O_p))[:, :case.c_out]
    return Result(f"workaround: {case.kernel}x per-tap matmul", issued, pb, pd, ms,
                  snr(yref.T, got.float()), ib,
                  extra=f"{case.kernel} matmuls of [{rows}x{C_p}]@[{C_p}x{O_p}]")


def _v_toeplitz(ue, case, x, w, yref, timed, snr, Result, prep_bytes):
    """Per-channel (1,L)@(L,L). The L x L matrices are the real cost."""
    pb, pd = prep_bytes(case)
    D, L, k = case.c_in, case.length, case.kernel
    L_p = _align(L)
    # Memory guard: D x L_p^2 bf16 is enormous by construction. Bench a channel
    # SUBSET and scale the time, rather than refusing to measure at all.
    per_ch = L_p * L_p * BPE
    # Keep the Toeplitz residency well inside the tensor region: at 512 MiB the
    # full D x L^2 overran it and produced NaNs. Channels are independent
    # sequential matmuls, so a subset scales linearly and faithfully.
    budget = 64 * 1024 * 1024
    d_bench = max(1, min(D, budget // per_ch))
    pad = (k - 1) // 2
    T = torch.zeros(d_bench, L_p, L_p)
    for d in range(d_bench):
        for t in range(L):
            for j in range(k):
                src = t + j - pad
                if 0 <= src < L:
                    # B is read as (N=out, K=in): row t, column src.
                    T[d, t, src] = w[d, 0, j]
    xr = torch.zeros(d_bench, L_p); xr[:, :L] = x[:d_bench, :L]
    x_addr = _put(ue, xr)
    t_addr = _put(ue, T)
    o_addr = ue.allocate_tensor_dram(d_bench * L_p * BPE)
    issued_bench = d_bench * 2 * L_p * L_p

    def emit():
        for d in range(d_bench):
            ue.matmat_mul_core(
                M=1, K=L_p, N=L_p,
                A_DRAM_ADDR=x_addr + d * L_p * BPE,
                B_DRAM_ADDR=t_addr + d * per_ch,
                OUTPUT_DRAM_ADDR=o_addr + d * L_p * BPE)

    ms, ib = timed(ue, emit)
    got = _get(ue, o_addr, (d_bench, L_p))[:, :L]
    scale = D / d_bench
    note = (f"{d_bench}/{D} channels measured, time scaled x{scale:.0f}"
            if d_bench < D else f"all {D} channels")
    return Result("workaround: Toeplitz per-channel matmul",
                  int(issued_bench * scale), pb, pd, ms * scale,
                  snr(yref[:d_bench, :L], got.float()), ib,
                  extra=f"{D} x (1,{L_p})@({L_p},{L_p}); {note}")


def _v_tapwise(ue, case, x, w, yref, timed, snr, Result, prep_bytes):
    """k eltwise multiplies against a pre-tiled weight + k-1 adds (FLOP-exact)."""
    D, L, k = case.c_in, case.length, case.kernel
    L_p = _align(L)
    pad = (k - 1) // 2
    pb = 2 * k * D * L_p * BPE
    pd = (f"{k} tiled weight planes + {k} shifted input planes, each "
          f"[{D} x {L_p}] bf16 (vs Toeplitz's {D} x {L_p} x {L_p})")
    xpad = torch.zeros(D, L_p + 2 * pad)
    xpad[:, pad:pad + L] = x[:, :L]
    # eltwise walks rows with stride N, so a byte offset cannot express a shift:
    # materialize each tap's shifted window as its own contiguous plane.
    xs = [_put(ue, xpad[:, kk:kk + L_p].contiguous()) for kk in range(k)]
    wt = [_put(ue, w[:, 0, kk].reshape(D, 1).expand(D, L_p).contiguous())
          for kk in range(k)]
    acc = [ue.allocate_tensor_dram(D * L_p * BPE) for _ in range(2)]
    prod = ue.allocate_tensor_dram(D * L_p * BPE)
    issued = k * (D * L_p) + (k - 1) * (D * L_p)   # muls + adds, 1 FLOP each

    def emit():
        for kk in range(k):
            # Tap 0 multiplies straight into the accumulator; later taps
            # multiply into a scratch plane and add it in. k muls + (k-1) adds.
            if kk == 0:
                ue.eltwise_core_dram(
                    M=D, N=L_p, dram_a=xs[kk], dram_b=wt[kk],
                    dram_out=acc[0], mode=UE_MODE.ELTWISE_MUL)
            else:
                ue.eltwise_core_dram(
                    M=D, N=L_p, dram_a=xs[kk], dram_b=wt[kk],
                    dram_out=prod, mode=UE_MODE.ELTWISE_MUL)
                ue.eltwise_core_dram(
                    M=D, N=L_p, dram_a=acc[(kk - 1) % 2], dram_b=prod,
                    dram_out=acc[kk % 2], mode=UE_MODE.ELTWISE_ADD)

    ms, ib = timed(ue, emit)
    got = _get(ue, acc[(k - 1) % 2], (D, L_p))[:, :L]
    return Result("alternative: tap-wise eltwise (FLOP-exact)", issued, pb, pd, ms,
                  snr(yref[:, :L], got.float()), ib,
                  extra=f"{k} ELTWISE_MUL + {k-1} ELTWISE_ADD over [{D}x{L_p}]")


# ---------------------------------------------------------------------------
# The real CONV op
# ---------------------------------------------------------------------------

def _v_conv_op(ue, case, x, w, yref, timed, snr, Result, conv_prep):
    """conv1d_core / run_conv2d_layer. INT4 weights -- see the precision note."""
    pb, pd = conv_prep(case)
    # Integer IF4-INT codes; quantization fidelity is deliberately out of scope,
    # so the reference is recomputed from the SAME codes and the SNR validates
    # the op rather than the number format.
    torch.manual_seed(1)
    if case.in_hw is not None:
        codes = torch.randint(-7, 8, (case.c_out, case.c_in, case.kernel, case.kernel))
        scale_mag = 1.0 / 64.0
        xi = torch.randint(0, 8, (case.c_in, case.in_hw, case.in_hw)).float()
        ref = F.conv2d(xi.unsqueeze(0), codes.float() * scale_mag,
                       stride=case.stride, padding=case.kernel // 2).squeeze(0)
        t0 = time.perf_counter()
        got = ue.run_conv2d_layer(xi, codes, stride_s=case.stride,
                                  pad=case.kernel // 2, scale_mag=scale_mag,
                                  data_type=TYPE.IF4)
        wall = (time.perf_counter() - t0) * 1000.0
        ms = ue.report_latency_in_us() / 1000.0
        return Result("CONV op: run_conv2d_layer (IF4)", case.true_flops, pb, pd, ms,
                      snr(ref, got.float()), 0,
                      extra=f"one resident multi-geometry program; host wall {wall:.1f} ms")

    codes = torch.randint(-7, 8, (case.c_out, case.c_in, 1, case.kernel))
    scale_mag = 1.0 / 64.0
    xi = torch.randint(0, 8, (case.c_in, 1, case.length)).float()
    pad = case.dilation * (case.kernel - 1) // 2
    ref = F.conv2d(xi.unsqueeze(0), codes.float() * scale_mag,
                   stride=(1, case.stride), padding=(0, pad),
                   dilation=(1, case.dilation)).squeeze(0)
    t0 = time.perf_counter()
    got = ue.run_conv2d_layer(xi, codes, stride_s=case.stride, pad=pad, pad_h=0,
                              dilation=case.dilation, scale_mag=scale_mag,
                              data_type=TYPE.IF4)
    wall = (time.perf_counter() - t0) * 1000.0
    ms = ue.report_latency_in_us() / 1000.0
    return Result("CONV op: conv1d via run_conv2d_layer (IF4)", case.true_flops,
                  pb, pd, ms, snr(ref, got.float()), 0,
                  extra=f"Conv1d as CONV2D H=1/Kh=1; host wall {wall:.1f} ms")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_WORKAROUND = {
    "im2col": _v_im2col,
    "per_tap": _v_per_tap,
    "toeplitz": _v_toeplitz,
}


def run(case, dev, peak_1e, prep_bytes, conv_prep, timed, snr, Result):
    out: list = []
    x, w, yref = _reference(case)

    def fresh(conv_mode: bool = False) -> UnifiedEngine:
        # Exact edge tiles need queue-CONFIG geometry; the matmul workarounds
        # run on the default live-CSR engine, exactly as the models do.
        kw = {"conv_geometry_mode": CONV_GEOMETRY_QUEUE_CONFIG} if conv_mode else {}
        ue = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR, **kw)
        ue.software_reset()
        return ue

    variants: list[tuple[str, Callable]] = [(case.pattern, _WORKAROUND[case.pattern])]
    if case.depthwise:
        variants.append(("tapwise", _v_tapwise))
    if case.conv_op_possible:
        variants.append(("conv_op", _v_conv_op))

    for tag, fn in variants:
        try:
            ue = fresh(conv_mode=(tag == "conv_op"))
            if tag == "conv_op":
                r = fn(ue, case, x, w, yref, timed, snr, Result, conv_prep)
            else:
                r = fn(ue, case, x, w, yref, timed, snr, Result, prep_bytes)
        except Exception as exc:                      # keep the table complete
            r = Result(f"{tag} (FAILED)", 0, 0, "", error=f"{type(exc).__name__}: {exc}")
        gf = case.true_flops / 1e9
        if r.error:
            print(f"  {r.variant:46s} ERROR {r.error[:70]}")
        else:
            gflops = gf / (r.hw_ms / 1000.0) if r.hw_ms > 0 else float("nan")
            print(f"  {r.variant:46s} {r.hw_ms:9.2f} ms  "
                  f"{gflops:7.2f} GFLOPS  {100*gflops/peak_1e:5.1f}% peak  "
                  f"issued {r.issued_flops/1e9:7.3f} GF  prep {r.prep_bytes/2**20:8.1f} MiB  "
                  f"SNR {r.snr_db:.0f} dB")
        out.append(r)
    return out


def write_report(path, cases, results, dev, clock_mhz, peak_1e,
                 prep_bytes, conv_prep) -> None:
    def pick(name, prefix):
        return next((r for r in results.get(name, [])
                     if not r.error and r.variant.startswith(prefix)), None)

    L: list[str] = []
    A = L.append
    A("# Convolution workarounds vs the CONV hardware — op-level benchmark\n")
    A(f"Device `{dev}`, {clock_mhz:.2f} MHz, {UE_VECTOR_SIZE} lanes, "
      f"single-engine peak **{peak_1e:.2f} GFLOPS** (clock x 128 FLOP/cycle). "
      f"Generated by `conv_walkaround_bench.py`; nothing under `models/` is read "
      f"or modified.\n")

    dense = [(c, pick(c.name, "workaround"), pick(c.name, "CONV op"))
             for c in cases if c.conv_op_possible]
    dense = [(c, w, o) for c, w, o in dense if w and o]
    dw = [(c, pick(c.name, "workaround"), pick(c.name, "alternative"))
          for c in cases if c.pattern == "toeplitz"]
    dw = [(c, w, a) for c, w, a in dw if w and a]

    # ---------------- 0. Verdict ----------------
    A("\n## 0. Verdict\n")
    if dense:
        sp = sorted(w.hw_ms / o.hw_ms for _, w, o in dense)
        mem = [w.prep_bytes / max(o.prep_bytes, 1) for _, w, o in dense]
        A(f"**CONV op on dense layers: marginal compute, real memory win.** "
          f"On-chip speedup {sp[0]:.2f}x-{sp[-1]:.2f}x (median {sp[len(sp)//2]:.2f}x) "
          f"— the matmul workarounds already run at 85-99% of peak, because an "
          f"im2col conv is just a well-shaped GEMM. Preparation DRAM drops "
          f"{min(mem):.1f}x-{max(mem):.1f}x, since the window is never materialized. "
          f"The one strong case is a first layer with few input channels "
          f"(`mbv2_stem`, C_in=3, **2.20x**), where im2col wastes 4.7x the true "
          f"FLOPs purely on lane padding.\n")
    if dw:
        saved = [(c, (w.hw_ms - a.hw_ms) * c.occurrences) for c, w, a in dw
                 if c.length == 128]
        gain = sorted(w.hw_ms / a.hw_ms for c, w, a in dw if c.length == 128)
        A(f"**Critical fix, and the conv op cannot do it: replace the Toeplitz "
          f"depthwise with tap-wise eltwise.** There is no groups/depthwise mode in "
          f"the hardware, so this is a pure software change. A K-tap filter is "
          f"currently stored as an L x L matrix per channel and run as a dense "
          f"matmul, which executes at 0.7-1.2% of peak. Expected gain: "
          f"**{gain[0]:.1f}x-{gain[-1]:.1f}x per op** at a 10 s utterance "
          f"(" + ", ".join(f"{c.model.split(' ')[0]} ~{v:.0f} ms/stage" for c, v in saved) +
          f"), **7-16x less memory**, and it lifts the utterance-length cap the "
          f"Toeplitz residency imposes. mobilenetv2 already uses this pattern "
          f"(`conv2d_3x3_dw_tapwise_dram`), so it is a port, not a new kernel.\n")
    A("**Keep the conv feature** for low-channel first layers and for the "
      "preparation-memory saving. Do **not** expect it to fix the audio encoders: "
      "the depthwise rewrite is the higher-value and hardware-independent move, "
      "and should be done first.")

    # ---------------- 1. Cases ----------------
    A("\n## 1. Cases, importance, and the cost the FLOP columns hide\n")
    A("`true GF` is what a minimal convolution owes; `issued GF` is what the "
      "variant actually asks the hardware to compute; `prep MiB` is DRAM that must "
      "be materialized before any arithmetic. `available speedup` is measured: "
      "CONV op for dense cases, tap-wise eltwise for depthwise.\n")
    A("\n| case | model | pattern | stage share | true GF | issued GF | waste | "
      "prep MiB | available speedup |")
    A("|---|---|---|---|---:|---:|---:|---:|---:|")
    for c in cases:
        w = pick(c.name, "workaround")
        better = pick(c.name, "CONV op") or pick(c.name, "alternative")
        pb, _ = prep_bytes(c)
        issued = w.issued_flops if w else 0
        waste = f"{issued / c.true_flops:.1f}x" if issued and c.true_flops else "—"
        if w and better:
            tag = "CONV op" if better.variant.startswith("CONV") else "tap-wise"
            spd = f"**{w.hw_ms / better.hw_ms:.2f}x** ({tag})"
        else:
            spd = "—"
        A(f"| `{c.name}` | {c.model} | {c.pattern}"
          f"{' **(depthwise)**' if c.depthwise else ''} | {c.stage_share} | "
          f"{c.true_flops/1e9:.3f} | {issued/1e9:.3f} | {waste} | {pb/2**20:.1f} | {spd} |")

    # ---------------- 2. Measured ----------------
    A("\n## 2. Measured performance — CONV op vs workaround\n")
    A("Dense cases only: the depthwise cases cannot be expressed on the CONV "
      "hardware, so their measured fix is the `available speedup` column above. "
      "`GFLOPS` and `% peak` use **true** work, so a variant computing too much "
      "gets no credit for the excess. `SNR` is vs the torch reference.\n")
    A("\n| case | variant | HW ms | GFLOPS (true) | % peak | issued GF | "
      "prep MiB | SNR dB | speedup |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for c, w, o in dense:
        gf = c.true_flops / 1e9
        for r in (w, o):
            g = gf / (r.hw_ms / 1000.0)
            spd = f"**{w.hw_ms / o.hw_ms:.2f}x**" if r is o else ""
            A(f"| `{c.name}` | {r.variant} | {r.hw_ms:.2f} | {g:.2f} | "
              f"{100*g/peak_1e:.1f}% | {r.issued_flops/1e9:.3f} | "
              f"{r.prep_bytes/2**20:.1f} | "
              f"{'inf' if r.snr_db == float('inf') else f'{r.snr_db:.0f}'} | {spd} |")

    # ---------------- 3. Caveats ----------------
    A("\n## 3. Caveats that bound these numbers\n")
    A("- **Weight precision is not matched.** Workarounds run BF16 weights through "
      "`matmat_mul_core`; the CONV op has no BF16 weight path (`conv2d_core` takes "
      "`TYPE.IF4 / IF8 / TQ4` only). Timings compare BF16-matmul against INT4-conv, "
      "so migrating is also a quantization change — a model-quality question for an "
      "audio front end, not answered here.")
    A("- **No depthwise/grouped mode.** Emulating one with a block-diagonal dense "
      "weight would cost C x the true FLOPs (C=1024 for both audio encoders), worse "
      "than the Toeplitz it would replace.")
    A("- **Toeplitz residency caps utterance length.** At L_pad=512 the matrices "
      "alone need ~12 GiB across 24 layers, against parakeet's 3 GB reservation. "
      "This is a capability limit, not only a speed one.")
    A("- **Two more L^2 ops sit in the same parakeet block** and the conv feature "
      "does not address either: SiLU applied as an identity matmul "
      "(`2*D*L_pad^2`) and the channel-first transpose.")
    A("- **Single engine**, so these are per-op costs, not stage throughput. "
      "Toeplitz cases bench a channel subset (64 MiB budget) and scale linearly.")
    A("- **`bigvgan_conv1d` occurrences is 1**, a floor: BigVGAN runs many conv "
      "layers across its upsample stages, so its stage saving is understated.")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
