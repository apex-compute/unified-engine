#!/usr/bin/env python3
"""Per-stage DRAM traffic + FLOP accounting for pi0.5, by instrumenting the emitters.

pi05_test already reports FLOPs per stage (`_vision_flops` / `_prefix_flops` /
`_denoise_flops`). What it has never reported is BYTES, and without bytes there is
no way to say whether a stage is compute-bound or memory-bound -- which is the
whole question when twelve engines share two ~9.6 GB/s memory ports.

This module wraps the op emitters (matmul, the norms, RoPE, attention, the URAM
round-trip copies) so that every op the compiler emits is recorded with its shape,
the engine it was emitted on, and the stage that was open at the time. Nothing is
modelled from the config: the record is whatever the compiler actually issued, so
per-head V^T rebuilds, staging copies and re-streamed weight tiles are all counted
where a from-the-manifest estimate would miss them.

Usage:

    python models/pi05/utility/pi05_traffic.py --engines max

which runs the ordinary pi05_test main() with instrumentation installed and
prints the traffic report at the end.
"""
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import user_dma_core
from user_dma_core import UnifiedEngine, URAM_NEAR_FULL_ELEMENTS, UE_VECTOR_SIZE, TYPE

# ---------------------------------------------------------------- state ----
_stage = "init"
_records = []          # (stage, engine_idx, op, flops, dram_read, dram_write, detail)
_eng_idx = {}          # id(ue) -> engine index


def _eidx(ue):
    return _eng_idx.get(id(ue), 0)


def _regval(ue, reg, fallback):
    """Value an ISA register was last ADD_SET to, or `fallback`.

    Needed because the compile-time M handed to an emitter is NOT always the M
    the hardware runs. The denoise stage is the case that matters: it passes the
    PHYSICAL row stride S=64 while priming gpr_M_reg with the trimmed
    AE_ACTION_HORIZON=10 (AE_TRIM_PAD_ROWS), so believing the literal
    over-counts every row-dependent term by 6.4x."""
    if reg is None:
        return fallback
    v = getattr(ue, "_prof_regs", {}).get(reg)
    return fallback if v is None else v


def _rec(ue, op, flops=0, read=0, write=0, **detail):
    _records.append((_stage, _eidx(ue), op, float(flops), float(read), float(write), detail))


# ------------------------------------------------------------ byte model ----
def _b_bytes_per_elem(is_quant, data_type):
    """Bytes of DRAM per element of the weight matrix B, scales included.

    q4_64 stores one bf16 scale per 64-element block along K, so IF4 costs
    0.5 + 2/64 bytes per weight, not 0.5. At 16384-wide MLP weights that 6%
    is tens of MB per prefix pass -- worth carrying exactly."""
    if not is_quant:
        return 2.0
    if data_type == TYPE.IF4:
        return 0.5 + 2.0 / 64
    if data_type == TYPE.IF8:
        return 1.0 + 2.0 / 64
    return 2.0


def _matmul_tiling(M, K, N):
    """Reproduce matmat_mul_core_dynamic's tiling so re-streamed weights are counted.

    The kernel loops M-tiles on the outside and N-strips on the inside, holding
    m_take rows of A and m_take x N_chunk of output in URAM at once. A is therefore
    read once per pass, but the WHOLE of B is re-streamed for every M-tile."""
    if K <= 0 or N <= 0:
        return 1, 1
    kr = max(1, K // UE_VECTOR_SIZE)
    n_chunk = min((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE * UE_VECTOR_SIZE,
                  (4095 // kr) * UE_VECTOR_SIZE)
    if n_chunk < UE_VECTOR_SIZE:          # sub-64 fallback path
        n_chunk = max(16, min(32, n_chunk) // 16 * 16)
    n_chunk = max(1, min(n_chunk, N))
    m_chunk = max(1, URAM_NEAR_FULL_ELEMENTS // (K + n_chunk))
    return math.ceil(M / m_chunk), math.ceil(N / n_chunk)


def _install(cls):
    """Wrap the op emitters on UnifiedEngine. Every wrapper records and delegates;
    none of them change what is emitted."""
    orig = {}

    def wrap(name, fn):
        orig[name] = getattr(cls, name)
        setattr(cls, name, fn)

    # ---- shadow the ISA register file so gpr_M_reg can be resolved ----------
    _set = cls.generate_instruction_add_set

    def generate_instruction_add_set(self, dst_reg_idx, immediate_value=0, *a, **kw):
        if not hasattr(self, "_prof_regs"):
            self._prof_regs = {}
        self._prof_regs[dst_reg_idx] = immediate_value
        return _set(self, dst_reg_idx, immediate_value, *a, **kw)

    wrap("generate_instruction_add_set", generate_instruction_add_set)

    # ---- matmul -------------------------------------------------------------
    _mm = cls.matmat_mul_core

    def matmat_mul_core(self, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                        softmax_enable=False, C_DRAM_ADDR=None, bias_mode="broadcast_N",
                        is_B_quantized=False, data_type=None, **kw):
        # rM/rK/rN are the RUNTIME dims, for accounting only. The literals M/K/N
        # are what the emitter must still receive -- passing the resolved value
        # through would change the compiled program, not just how it is counted.
        rM = _regval(self, kw.get("gpr_M_reg"), M)
        rK = _regval(self, kw.get("gpr_K_reg"), K)
        rN = _regval(self, kw.get("gpr_N_reg"), N)
        mtiles, _ = _matmul_tiling(rM, rK, rN)
        be = _b_bytes_per_elem(is_B_quantized, data_type)
        read = rM * rK * 2 + mtiles * rN * rK * be + (rN * 2 if C_DRAM_ADDR is not None else 0)
        write = 0 if kw.get("write_back_disable") else rM * rN * 2
        _rec(self, "matmul", flops=2 * rM * rK * rN, read=read, write=write,
             M=rM, K=rK, N=rN, quant=bool(is_B_quantized),
             dt=(data_type.name if hasattr(data_type, "name") else str(data_type)),
             mtiles=mtiles)
        return _mm(self, M, K, N, A_DRAM_ADDR, B_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                   softmax_enable, C_DRAM_ADDR, bias_mode, is_B_quantized, data_type, **kw)

    wrap("matmat_mul_core", matmat_mul_core)

    # ---- norms --------------------------------------------------------------
    _rms = cls.rms_norm_core_dram

    def rms_norm_core_dram(self, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR, GAMMA_DRAM_ADDR, **kw):
        rM = _regval(self, kw.get("gpr_M_reg"), M)
        rN = _regval(self, kw.get("gpr_N_reg"), N)
        _rec(self, "rms_norm", flops=3 * rM * rN, read=rM * rN * 2 + rN * 2,
             write=rM * rN * 2, M=rM, N=rN)
        return _rms(self, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR, GAMMA_DRAM_ADDR, **kw)

    wrap("rms_norm_core_dram", rms_norm_core_dram)

    _ln = cls.layer_norm_core_dram

    def layer_norm_core_dram(self, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR, **kw):
        rM = _regval(self, kw.get("gpr_M_reg"), M)
        rN = _regval(self, kw.get("gpr_N_reg"), N)
        _rec(self, "layer_norm", flops=5 * rM * rN, read=rM * rN * 2 + 2 * rN * 2,
             write=rM * rN * 2, M=rM, N=rN)
        return _ln(self, M, N, A_DRAM_ADDR, OUTPUT_DRAM_ADDR, **kw)

    wrap("layer_norm_core_dram", layer_norm_core_dram)

    # ---- eltwise ------------------------------------------------------------
    _elt = cls.eltwise_core_dram

    def eltwise_core_dram(self, M, N, dram_a, dram_b, dram_out, mode, **kw):
        read = M * N * 2 * (2 if dram_b is not None else 1)
        _rec(self, "eltwise", flops=M * N, read=read, write=M * N * 2,
             M=M, N=N, mode=getattr(mode, "name", str(mode)))
        return _elt(self, M, N, dram_a, dram_b, dram_out, mode, **kw)

    wrap("eltwise_core_dram", eltwise_core_dram)

    # ---- RoPE ---------------------------------------------------------------
    _rope = cls.rope_hf_core_dram

    def rope_hf_core_dram(self, M, N, input_dram_addr, output_dram_addr,
                          cos_dram_addr, sin_dram_addr, **kw):
        rM = _regval(self, kw.get("gpr_M_reg"), M)
        rN = _regval(self, kw.get("gpr_N_reg"), N)
        # input + cos + sin in, rotated out.
        _rec(self, "rope", flops=6 * rM * rN, read=rM * rN * 2 * 3, write=rM * rN * 2,
             M=rM, N=rN)
        return _rope(self, M, N, input_dram_addr, output_dram_addr,
                     cos_dram_addr, sin_dram_addr, **kw)

    wrap("rope_hf_core_dram", rope_hf_core_dram)

    _ropeg = cls.rope_hf_core_dram_gqa

    def rope_hf_core_dram_gqa(self, M, group_size, N, input_dram_addr, output_dram_addr,
                              cos_dram_addr, sin_dram_addr, **kw):
        rM = _regval(self, kw.get("gpr_M_reg"), M)
        rN = _regval(self, kw.get("gpr_N_reg"), N)
        _rec(self, "rope", flops=6 * rM * rN, read=rM * rN * 2 * 3, write=rM * rN * 2,
             M=rM, N=rN)
        return _ropeg(self, M, group_size, N, input_dram_addr, output_dram_addr,
                      cos_dram_addr, sin_dram_addr, **kw)

    wrap("rope_hf_core_dram_gqa", rope_hf_core_dram_gqa)

    # ---- attention ----------------------------------------------------------
    _attn = cls.unified_attention_core_dynamic

    def unified_attention_core_dynamic(self, batch, aligned_seq_len, head_dim,
                                       Q_DRAM_ADDR, K_DRAM_ADDR, V_DRAM_ADDR,
                                       BIAS_DRAM_ADDR, OUTPUT_DRAM_ADDR,
                                       SCRATCH_DRAM_ADDR, **kw):
        B = _regval(self, kw.get("gpr_batch_reg"), batch)
        S, D = aligned_seq_len, head_dim
        flops = 2 * B * S * D * 2                       # QK^T then P@V
        # Q in, K and V in, bias in; scores out to scratch and back in; out.
        read = B * D * 2 + 2 * S * D * 2 + B * S * 2 + B * S * 2
        write = B * S * 2 + B * D * 2
        _rec(self, "attention", flops=flops, read=read, write=write,
             batch=B, kv=S, hd=D)
        return _attn(self, batch, aligned_seq_len, head_dim, Q_DRAM_ADDR, K_DRAM_ADDR,
                     V_DRAM_ADDR, BIAS_DRAM_ADDR, OUTPUT_DRAM_ADDR, SCRATCH_DRAM_ADDR, **kw)

    wrap("unified_attention_core_dynamic", unified_attention_core_dynamic)

    # ---- URAM round-trip copies (DRAM->URAM->DRAM staging) ------------------
    _r = cls.accelerator_memory_to_sram

    def accelerator_memory_to_sram(self, accelerator_dram_address, sram_address,
                                   element_size, **kw):
        nb = kw.get("memcpy_length_bytes")
        nb = element_size * 2 if nb is None else nb
        _rec(self, "copy_in", read=nb)
        return _r(self, accelerator_dram_address, sram_address, element_size, **kw)

    wrap("accelerator_memory_to_sram", accelerator_memory_to_sram)

    _w = cls.sram_to_accelerator_memory

    def sram_to_accelerator_memory(self, sram_address, accelerator_dram_address,
                                   element_size, **kw):
        nb = kw.get("memcpy_length_bytes")
        nb = element_size * 2 if nb is None else nb
        _rec(self, "copy_out", write=nb)
        return _w(self, sram_address, accelerator_dram_address, element_size, **kw)

    wrap("sram_to_accelerator_memory", sram_to_accelerator_memory)
    return orig


_exec_counts = collections.Counter()
def _label_stage(label):
    """Map a launch label onto a stage. The vision encoder does not go through
    _execute at all -- run_vision drives it with a bare start_execute_from_dram
    plus a heartbeat wait per image slot -- so both launch paths are hooked."""
    if label.startswith("vision slot") or label.startswith("encoder slot"):
        return "vision"
    return {"prefix": "prefix", "denoise_loop": "denoise"}.get(label)


def install(pi05_cls, sched_cls):
    """Instrument the emitters, tag engines with their index, and bracket the
    three compile entry points so every op lands in the right stage."""
    _install(UnifiedEngine)

    _sched_init = sched_cls.__init__

    def __init__(self, primary_ue, *a, **kw):
        _sched_init(self, primary_ue, *a, **kw)
        for i, e in enumerate(self.engines):
            _eng_idx[id(e)] = i

    sched_cls.__init__ = __init__

    # A compiled program is not necessarily run once. The vision encoder is
    # compiled ONCE and executed once per image slot (three for LIBERO), so the
    # emitted op list is a THIRD of the work the stage actually does. Count the
    # launches rather than assuming.
    _ex = pi05_cls._execute

    def _execute(self, prog_addr, label="execute", timeout=250.0):
        return _ex(self, prog_addr, label=label, timeout=timeout)

    pi05_cls._execute = _execute

    _wh = pi05_cls._wait_with_heartbeat

    def _wait_with_heartbeat(self, label, *a, **kw):
        st = _label_stage(label)
        if st:
            _exec_counts[st] += 1
        return _wh(self, label, *a, **kw)

    pi05_cls._wait_with_heartbeat = _wait_with_heartbeat

    for fn_name, stage in (("weight_init", "weight_init"),
                           ("weight_init_action_expert", "weight_init"),
                           ("tensor_init", "tensor_init"),
                           ("tensor_init_action_expert", "tensor_init"),
                           ("compile_encoder", "vision"),
                           ("compile_prefix", "prefix"),
                           ("compile_denoise_loop", "denoise")):
        f = getattr(pi05_cls, fn_name, None)
        if f is None:
            continue

        def mk(f, stage):
            def wrapper(self, *a, **kw):
                global _stage
                prev, _stage = _stage, stage
                try:
                    return f(self, *a, **kw)
                finally:
                    _stage = prev
            return wrapper

        setattr(pi05_cls, fn_name, mk(f, stage))


# --------------------------------------------------------------- report ----
# Measured on this board by pi05_bw_scaling.py: engines 0-7 share one memory
# port and engines 8-11 share a second, each port topping out near 9.6 GB/s --
# and a SINGLE engine already saturates its port. See that script for the sweep.
PORT_OF_ENGINE = lambda i: 0 if i < 8 else 1
PORT_GBS = 9.6


def report(peak_gflops_per_engine, n_engines, stage_seconds=None, out=sys.stdout):
    by_stage = collections.defaultdict(lambda: collections.defaultdict(float))
    by_stage_eng = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(float)))
    by_stage_op = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(float)))

    mult = {st: max(1, n) for st, n in _exec_counts.items()}
    for stage, eng, op, fl, rd, wr, _d in _records:
        m = mult.get(stage, 1)
        fl, rd, wr = fl * m, rd * m, wr * m
        s = by_stage[stage]
        s["flops"] += fl; s["read"] += rd; s["write"] += wr; s["ops"] += 1
        e = by_stage_eng[stage][eng]
        e["flops"] += fl; e["read"] += rd; e["write"] += wr; e["ops"] += 1
        o = by_stage_op[stage][op]
        o["flops"] += fl; o["read"] += rd; o["write"] += wr; o["ops"] += 1

    p = lambda *a: print(*a, file=out)
    p("\n" + "=" * 100)
    p("pi0.5 per-stage DRAM traffic and arithmetic intensity")
    p("=" * 100)
    p(f"compute ceiling : {peak_gflops_per_engine:.1f} GFLOP/s per engine "
      f"x {n_engines} = {peak_gflops_per_engine * n_engines:.1f} GFLOP/s")
    p(f"memory ceiling  : {PORT_GBS:.1f} GB/s per port x 2 ports = {2*PORT_GBS:.1f} GB/s "
      f"(engines 0-7 on port 0, 8-11 on port 1)")
    p("")

    for stage in ("vision", "prefix", "denoise", "weight_init", "tensor_init", "init"):
        s = by_stage.get(stage)
        if not s or (s["flops"] == 0 and s["read"] == 0 and s["write"] == 0):
            continue
        traffic = s["read"] + s["write"]
        ai = s["flops"] / traffic if traffic else float("inf")

        engs = sorted(by_stage_eng[stage])
        # Port-aware time floors: each port carries the traffic of the engines on
        # it, and each engine's compute is capped by its own ALU.
        port_bytes = {0: 0.0, 1: 0.0}
        for e in engs:
            ee = by_stage_eng[stage][e]
            port_bytes[PORT_OF_ENGINE(e)] += ee["read"] + ee["write"]
        t_mem = max(port_bytes[0], port_bytes[1]) / (PORT_GBS * 1e9)
        t_cmp = max((by_stage_eng[stage][e]["flops"] for e in engs), default=0.0) \
            / (peak_gflops_per_engine * 1e9)

        p(f"--- {stage} " + "-" * (96 - len(stage)))
        p(f"  engines used     : {len(engs)}  {engs}")
        p(f"  program launches : {mult.get(stage, 1)}  "
          f"(emitted ops are counted once per launch)")
        p(f"  ops emitted      : {int(s['ops']):,}")
        p(f"  FLOPs            : {s['flops']/1e9:10.2f} G")
        p(f"  DRAM read        : {s['read']/1e6:10.1f} MB")
        p(f"  DRAM write       : {s['write']/1e6:10.1f} MB")
        p(f"  arithmetic inten.: {ai:10.2f} FLOP/byte")
        p(f"  port 0 traffic   : {port_bytes[0]/1e6:10.1f} MB   "
          f"port 1 traffic: {port_bytes[1]/1e6:.1f} MB")
        p(f"  floor (memory)   : {t_mem*1e3:10.1f} ms   <- slowest port")
        p(f"  floor (compute)  : {t_cmp*1e3:10.1f} ms   <- busiest engine")
        bound = "MEMORY" if t_mem > t_cmp else "COMPUTE"
        p(f"  => {bound}-bound by {max(t_mem,t_cmp)/max(min(t_mem,t_cmp),1e-12):.2f}x")
        if stage_seconds and stage in stage_seconds:
            meas = stage_seconds[stage]
            p(f"  MEASURED         : {meas*1e3:10.1f} ms  "
              f"({meas/max(t_mem,t_cmp,1e-12):.1f}x the roofline floor -> "
              f"{100*max(t_mem,t_cmp)/meas:.1f}% of the achievable ceiling)")
            p(f"     achieved       : {s['flops']/meas/1e9:.1f} GFLOP/s, "
              f"{traffic/meas/1e9:.2f} GB/s")

        # per-op breakdown, biggest traffic first
        p(f"  {'op':<12}{'count':>8}{'GFLOP':>12}{'read MB':>12}{'write MB':>12}{'FLOP/byte':>12}")
        for op, o in sorted(by_stage_op[stage].items(),
                            key=lambda kv: -(kv[1]["read"] + kv[1]["write"])):
            t = o["read"] + o["write"]
            p(f"  {op:<12}{int(o['ops']):>8}{o['flops']/1e9:>12.2f}"
              f"{o['read']/1e6:>12.1f}{o['write']/1e6:>12.1f}"
              f"{(o['flops']/t if t else 0):>12.2f}")

        # load balance across engines
        if len(engs) > 1:
            tot = [by_stage_eng[stage][e]["read"] + by_stage_eng[stage][e]["write"]
                   for e in engs]
            fl = [by_stage_eng[stage][e]["flops"] for e in engs]
            p(f"  per-engine traffic MB: " +
              " ".join(f"{t/1e6:.0f}" for t in tot))
            p(f"  per-engine GFLOP     : " +
              " ".join(f"{f/1e9:.1f}" for f in fl))
            if max(fl) > 0:
                p(f"  compute imbalance    : busiest/lightest = {max(fl)/max(min(fl),1e-9):.2f}x"
                  f"   (a barrier makes every engine pay the busiest)")
        p("")


def main():
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    import pi05_test
    import multi_engine_shard

    install(pi05_test.Pi05Libero_UnifiedEngine, multi_engine_shard.MultiEngineScheduler)

    stage_seconds = {}
    _orig_rg = pi05_test.Pi05Libero_UnifiedEngine._report_gflops

    def _report_gflops(self, label, flops, seconds):
        key = label.split()[0]
        stage_seconds[{"vision": "vision", "prefix": "prefix",
                       "denoise": "denoise"}.get(key, key)] = seconds
        return _orig_rg(self, label, flops, seconds)

    pi05_test.Pi05Libero_UnifiedEngine._report_gflops = _report_gflops

    cls = pi05_test.Pi05Libero_UnifiedEngine
    peak = cls.DEVICE_PEAK_GFLOPS or (cls.MACS_PER_CYCLE * 2 / (cls.CYCLE_NS * 1e-9) / 1e9)

    try:
        pi05_test.main()
    finally:
        ne = max([cls.NUM_ENGINES] + [getattr(cls, f"{s}_NUM_ENGINES", 1) or 1
                                      for s in ("VIS", "PREFIX", "DENOISE")])
        report(peak, ne, stage_seconds)
        path = os.path.join(os.getcwd(), "pi05_traffic_report.txt")
        with open(path, "w") as f:
            report(peak, ne, stage_seconds, out=f)
        print(f"\n[traffic] report written to {path}")


if __name__ == "__main__":
    main()
