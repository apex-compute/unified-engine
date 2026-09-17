#!/usr/bin/env python3
"""One Qwen2.5-Omni MLP layer, row-sharded vs tensor-parallel, on real hardware.

WHY THIS EXISTS
    Moving the Omni map to 1 GiB private windows per core only fits if the
    prefill MLP stops reading one SHARED copy of gate/up/down and reads a
    private column shard instead (the shared copy is 2889 MiB of the 3655 MiB
    decoder; with it resident there is no contiguous room left for the 112 MiB
    KV cache). That change is worth ~79% of the decoder weights, but it swaps a
    per-engine in-lane accumulation for a CROSS-ENGINE reduce_add, and the cost
    of that trade is not predictable from the shapes alone. So it gets measured
    on one layer before 28 layers are converted.

THE TWO PATHS, AND WHY THEY ARE TRANSPOSES OF EACH OTHER
    Today's prefill splits the SEQUENCE: engine e owns rows [e*64, e*64+64) and
    runs all 8 down-K lanes itself, reading every lane's weights from the shared
    blob by address arithmetic (qwen2.5_vl_3b_lm.py:1861). Tensor-parallel
    splits the WEIGHTS instead: engine e owns lane e alone and runs it for ALL
    512 rows. Identical arithmetic, transposed assignment:

        rowshard   engine e: rows[e], lanes 0..7    weights SHARED
        tp         engine e: rows 0..511, lane e    weights PRIVATE

    The elementwise SiLU-multiply never leaves its lane in either path. The
    difference is entirely in `down`, which contracts over the lane dimension:
    rowshard accumulates its 8 lane partials locally, tp must reduce 8 partials
    ACROSS engines (barrier + 7 adds of [M, H] + barrier).

WHAT IS MEASURED
    HW latency for one layer at the real prefill tile (M=512=8x64, H=3584,
    MLP=18944, IF4 gate/up/down), plus a numerics check of both paths against a
    CPU reference built from the DEQUANTIZED staged blocks -- comparing against
    the pre-quantization tensor would hide compute error under quant error.

    The arena is the PROPOSED 1 GiB geometry, so this also exercises the new
    map and PrivateArena.alloc_shared on hardware rather than only in
    simulation.

    python models/qwen2.5_omni_7b/qwen2.5_omni_7b_mlp_tp_test.py --mode both
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

import user_dma_core
from multi_engine_shard import MultiEngineScheduler, PrivateArena
from quant_lib import dequantize_if4 as qs_dequantize_if4
from quant_lib import quantize as qs_quantize
from user_dma_core import (DMA_DEVICE_H2C, TYPE, UE_MODE, UE_VECTOR_SIZE,
                           UnifiedEngine)

MiB = 2**20
BPE = 2

# Qwen2.5-Omni-7B Thinker geometry (qwen2.5_omni_7b_config.json).
H = 3584
MLP = 18944

# The proposed map: 8 x 1 GiB windows tiling the whole 8 GiB device, ISA and
# tensor scratch INSIDE each window, shared data carved from the window tails.
ENGINES = 8
WINDOW_BYTES = 0x4000_0000
PRIVATE_ISA_BYTES = 32 * MiB
PRIVATE_TENSOR_BYTES = 8 * MiB
ENGINE_BASE_STRIDE = 0x00010000


def build_dual(num_engines: int):
    """TWO 512 MB-aligned windows per core instead of one 1 GiB window.

    Same 1 GiB per core and the same 8 GiB total, but each half keeps the
    512 MB alignment the DRAM controller interleaves on: core i owns
    [i*512MB, +512MB) in the low arena and [4GiB + i*512MB, +512MB) in the high
    one. Weights are split across the pair.
    """
    half = 0x2000_0000
    lo = PrivateArena(num_engines, arena_base=0, arena_bytes=num_engines * half,
                      isa_bytes=PRIVATE_ISA_BYTES, tensor_bytes=PRIVATE_TENSOR_BYTES)
    hi = PrivateArena(num_engines, arena_base=num_engines * half,
                      arena_bytes=num_engines * half,
                      isa_bytes=PRIVATE_ISA_BYTES, tensor_bytes=PRIVATE_TENSOR_BYTES)
    r0 = lo.region(0)
    primary = UnifiedEngine(
        BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
        params_dram_base=r0.weight_base,
        tensor_dram_base=r0.tensor_base,
        program_dram_base=r0.isa_base,
    )
    sched = MultiEngineScheduler(
        primary, num_engines=num_engines, engine_base_stride=ENGINE_BASE_STRIDE,
        arena=lo, barrier_margin_nops=32, allow_unaligned_rows=True,
        allow_more_than_two_engines=num_engines > 2)
    return primary, sched, lo, hi


def build(num_engines: int, window_bytes: int = WINDOW_BYTES, arena_base: int = 0):
    """Primary + scheduler over the proposed private arena."""
    arena = PrivateArena(
        num_engines,
        arena_base=arena_base,
        arena_bytes=num_engines * window_bytes,
        isa_bytes=PRIVATE_ISA_BYTES,
        tensor_bytes=PRIVATE_TENSOR_BYTES,
        verbose=True,
    )
    region0 = arena.region(0)
    primary = UnifiedEngine(
        BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
        params_dram_base=region0.weight_base,
        tensor_dram_base=region0.tensor_base,
        program_dram_base=region0.isa_base,
    )
    sched = MultiEngineScheduler(
        primary,
        num_engines=num_engines,
        engine_base_stride=ENGINE_BASE_STRIDE,
        arena=arena,
        barrier_margin_nops=32,
        # The model's own prefill runs 8 rows per engine for a one-block (64
        # global row) prompt -- _prefill_execution_rows() rounds to 64, not to
        # 64-per-engine -- so the benchmark must be able to do the same.
        allow_unaligned_rows=True,
        allow_more_than_two_engines=num_engines > 2,
    )
    return primary, sched, arena


def _q(block: torch.Tensor):
    """IF4-quantize an [N, K] weight block the way params.bin was built."""
    return qs_quantize("if4", block.contiguous(), block_size=UE_VECTOR_SIZE)


def make_weights(seed: int):
    """gate/up are [MLP, H]; down is [H, MLP]. Both stored N x K row-major."""
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(H)
    gate = (torch.rand(MLP, H, dtype=torch.bfloat16) * 2 - 1) * scale
    up = (torch.rand(MLP, H, dtype=torch.bfloat16) * 2 - 1) * scale
    down = (torch.rand(H, MLP, dtype=torch.bfloat16) * 2 - 1) / math.sqrt(MLP)
    return gate, up, down


def stage_rowshard(primary, arena, gate, up, down, lanes):
    """Baseline: ONE shared copy of each weight, in the shared pool.

    gate/up stay whole -- a down K-lane is an N-slice of gate/up, which is a
    contiguous row block of an [N, K] blob, so lanes are address arithmetic on
    the unsliced image (exactly what the model does today). `down` is the
    asymmetric one: a K-slice is a COLUMN slice of its rows, so its lanes must
    be pre-sliced host-side and uploaded as separate blobs.
    """
    LANE = MLP // lanes
    out = {"lane": LANE, "blocks": {}}
    for tag, w in (("gate", gate), ("up", up)):
        data_b, scale_b = _q(w)
        d = arena.alloc_shared(len(data_b), f"prefill.{tag}.data")
        primary.dma_write(DMA_DEVICE_H2C, d, data_b, len(data_b))
        s = arena.alloc_shared(len(scale_b), f"prefill.{tag}.scale")
        primary.dma_write(DMA_DEVICE_H2C, s, scale_b, len(scale_b))
        out[tag] = (d, s)
        out["blocks"][tag] = (data_b, scale_b)
    for lane in range(lanes):
        blk = down[:, lane * LANE:(lane + 1) * LANE]
        data_b, scale_b = _q(blk)
        d = arena.alloc_shared(len(data_b), f"prefill.down.l{lane}.data")
        primary.dma_write(DMA_DEVICE_H2C, d, data_b, len(data_b))
        s = arena.alloc_shared(len(scale_b), f"prefill.down.l{lane}.scale")
        primary.dma_write(DMA_DEVICE_H2C, s, scale_b, len(scale_b))
        out[f"down{lane}"] = (d, s)
        out["blocks"][f"down{lane}"] = (data_b, scale_b)
    return out


_DOWN_ARENA = [None]


def stage_tp(primary, arena, gate, up, down, num_engines):
    """Tensor-parallel: engine e holds ONLY lane e, in its own private window."""
    LANE = MLP // num_engines
    out = {"lane": LANE, "blocks": {}}
    for e in range(num_engines):
        for tag, w in (("gate", gate), ("up", up)):
            blk = w[e * LANE:(e + 1) * LANE, :]
            data_b, scale_b = _q(blk)
            d = arena.alloc_weights(e, len(data_b), f"mlp.{tag}.shard")
            primary.dma_write(DMA_DEVICE_H2C, d, data_b, len(data_b))
            s = arena.alloc_weights(e, len(scale_b), f"mlp.{tag}.scale")
            primary.dma_write(DMA_DEVICE_H2C, s, scale_b, len(scale_b))
            out[(tag, e)] = (d, s)
            out["blocks"][(tag, e)] = (data_b, scale_b)
        blk = down[:, e * LANE:(e + 1) * LANE]
        data_b, scale_b = _q(blk)
        down_arena = _DOWN_ARENA[0] or arena
        d = down_arena.alloc_weights(e, len(data_b), "mlp.down.shard")
        primary.dma_write(DMA_DEVICE_H2C, d, data_b, len(data_b))
        s = down_arena.alloc_weights(e, len(scale_b), "mlp.down.scale")
        primary.dma_write(DMA_DEVICE_H2C, s, scale_b, len(scale_b))
        out[("down", e)] = (d, s)
        out["blocks"][("down", e)] = (data_b, scale_b)
    return out


def run_case(mode: str, num_engines: int, M: int, timeout_s: float, seed: int,
             window_bytes: int = WINDOW_BYTES, dual: bool = False,
             arena_base: int = 0):
    """One path, one layer. Returns a result dict; never raises on a hang."""
    gate, up, down = make_weights(seed)
    torch.manual_seed(seed + 1)
    a_host = torch.randn(M, H, dtype=torch.bfloat16) / math.sqrt(H)

    hi_arena = None
    if dual:
        primary, sched, arena, hi_arena = build_dual(num_engines)
    else:
        primary, sched, arena = build(num_engines, window_bytes, arena_base)
    sched.preclear_flags()
    lanes = num_engines
    LANE = MLP // lanes

    _DOWN_ARENA[0] = hi_arena
    staged = (stage_tp(primary, arena, gate, up, down, num_engines) if mode == "tp"
              else stage_rowshard(primary, arena, gate, up, down, lanes))

    # Shared activation, read by every engine in both paths.
    a_addr = arena.alloc_shared(M * H * BPE, "act.mlp_norm")
    primary.dma_to_accelerator_memory(a_addr, a_host)
    out_addr = arena.alloc_shared(M * H * BPE, "act.mlp_down")

    if mode == "tp":
        # Per-engine [M, LANE] gate/up/mult stay in lane; each engine's down
        # partial is a full [M, H] that the reduction sums.
        g_addr = [arena.alloc_shared(M * LANE * BPE, f"act.gate.e{e}") for e in range(num_engines)]
        u_addr = [arena.alloc_shared(M * LANE * BPE, f"act.up.e{e}") for e in range(num_engines)]
        m_addr = [arena.alloc_shared(M * LANE * BPE, f"act.mult.e{e}") for e in range(num_engines)]
        p_addr = [arena.alloc_shared(M * H * BPE, f"act.down_partial.e{e}") for e in range(num_engines)]
    else:
        # One [M, MLP] plane for gate/up/mult, lane l at + l*M*LANE*bpe, and a
        # single [M, H] accumulator -- the model's own prefill buffers.
        g_addr = arena.alloc_shared(M * MLP * BPE, "act.gate")
        u_addr = arena.alloc_shared(M * MLP * BPE, "act.up")
        m_addr = arena.alloc_shared(M * MLP * BPE, "act.mult")
        p_addr = arena.alloc_shared(M * H * BPE, "act.down_partial")

    print(arena.describe_shared())
    arena.verify(verbose=False)

    primary.start_capture()
    sched.begin_program()
    m_regs = [primary.alloc_isa_reg()] + [w.alloc_isa_reg() for w in sched.workers]

    flops = 0
    if mode == "tp":
        # ---- tensor-parallel: every engine runs ALL rows for its own lane ----
        ctxs = sched.begin_col_sharded(MLP)
        for c in ctxs:
            e, ue = c.engine_idx, c.ue
            assert c.cols == LANE, f"col split gave {c.cols}, expected {LANE}"
            ue.generate_instruction_add_set(m_regs[e], M)
            for tag, dst in (("gate", g_addr), ("up", u_addr)):
                d, s = staged[(tag, e)]
                ue.matmat_mul_core(
                    M=M, K=H, N=LANE, A_DRAM_ADDR=a_addr,
                    B_DRAM_ADDR=d, SCALE_DRAM_ADDR=s, is_B_quantized=True,
                    data_type=TYPE.IF4, OUTPUT_DRAM_ADDR=dst[e],
                    silu_enable=(tag == "gate"), gpr_M_reg=m_regs[e])
                flops += 2 * M * H * LANE
            ue.eltwise_core_dram(
                M=M, N=LANE, dram_a=g_addr[e], dram_b=u_addr[e],
                dram_out=m_addr[e], mode=UE_MODE.ELTWISE_MUL,
                gpr_M_reg=m_regs[e])
            d, s = staged[("down", e)]
            ue.matmat_mul_core(
                M=M, K=LANE, N=H, A_DRAM_ADDR=m_addr[e],
                B_DRAM_ADDR=d, SCALE_DRAM_ADDR=s, is_B_quantized=True,
                data_type=TYPE.IF4, OUTPUT_DRAM_ADDR=p_addr[e],
                gpr_M_reg=m_regs[e])
            flops += 2 * M * LANE * H
        sched.end_sharded(join=True)
        # The one true cross-engine step. parallel=True spreads the adds by row
        # block instead of leaving them all on the primary.
        sched.reduce_add(p_addr, out_addr, M=M, N=H, parallel=True)
    else:
        # ---- row-shard baseline: engine e owns its rows, runs every lane ----
        ctxs = sched.begin_sharded(M)
        for c in ctxs:
            e, ue = c.engine_idx, c.ue
            r0, rn = c.row_offset, c.rows
            ue.generate_instruction_add_set(m_regs[e], rn)
            a_rows = a_addr + r0 * H * BPE
            for lane in range(lanes):
                lane_plane = M * LANE * BPE
                g = g_addr + lane * lane_plane + r0 * LANE * BPE
                u = u_addr + lane * lane_plane + r0 * LANE * BPE
                mu = m_addr + lane * lane_plane + r0 * LANE * BPE
                for tag, dst in (("gate", g), ("up", u)):
                    d, s = staged[tag]
                    # lane l of gate/up is a contiguous row block of [MLP, H]
                    ue.matmat_mul_core(
                        M=rn, K=H, N=LANE, A_DRAM_ADDR=a_rows,
                        B_DRAM_ADDR=d + lane * LANE * (H // 2),
                        SCALE_DRAM_ADDR=s + lane * LANE * (H // UE_VECTOR_SIZE) * BPE,
                        is_B_quantized=True, data_type=TYPE.IF4,
                        OUTPUT_DRAM_ADDR=dst,
                        silu_enable=(tag == "gate"), gpr_M_reg=m_regs[e])
                    flops += 2 * rn * H * LANE
                ue.eltwise_core_dram(
                    M=rn, N=LANE, dram_a=g, dram_b=u, dram_out=mu,
                    mode=UE_MODE.ELTWISE_MUL, gpr_M_reg=m_regs[e])
                d, s = staged[f"down{lane}"]
                first = (lane == 0)
                dst = (out_addr if first else p_addr) + r0 * H * BPE
                ue.matmat_mul_core(
                    M=rn, K=LANE, N=H, A_DRAM_ADDR=mu,
                    B_DRAM_ADDR=d, SCALE_DRAM_ADDR=s, is_B_quantized=True,
                    data_type=TYPE.IF4, OUTPUT_DRAM_ADDR=dst,
                    gpr_M_reg=m_regs[e])
                flops += 2 * rn * LANE * H
                if not first:
                    # Lane 0 wrote the accumulator; later lanes sum in.
                    ue.eltwise_core_dram(
                        M=rn, N=H, dram_a=out_addr + r0 * H * BPE,
                        dram_b=p_addr + r0 * H * BPE,
                        dram_out=out_addr + r0 * H * BPE,
                        mode=UE_MODE.ELTWISE_ADD, gpr_M_reg=m_regs[e])
        sched.end_sharded(join=True)

    worker_addrs = sched.finalize()
    primary.generate_instruction_halt()
    primary.stop_capture()
    prog_addr = primary.get_program_dram_addr()
    primary.write_captured_instructions_to_dram(prog_addr)
    master_bytes = primary.get_capture_instruction_size_bytes()
    worker_bytes = sched.worker_program_bytes() // max(1, len(sched.workers))

    sched.start_workers(worker_addrs)
    primary.start_execute_from_dram(prog_addr)
    t0 = time.perf_counter()
    try:
        primary.wait_queue(timeout_s)
        hung = False
    except Exception:
        hung = True
    wall = time.perf_counter() - t0

    res = {"mode": mode, "hung": hung, "wall_s": wall, "flops": flops,
           "latency_us": None if hung else primary.report_latency_in_us(),
           "master_MiB": master_bytes / MiB, "worker_MiB": worker_bytes / MiB,
           "private_MiB": max(arena.usage()) / MiB,
           "shared_MiB": sum(arena.shared_usage()) / MiB,
           "snr_db": None}
    if not hung:
        got = primary.dma_from_accelerator_memory(out_addr, (M, H))
        blocks = staged["blocks"]
        if mode == "tp":
            g_eff = torch.cat([qs_dequantize_if4(*blocks[("gate", e)], N=LANE, K=H,
                                                 block_size=UE_VECTOR_SIZE)
                               for e in range(num_engines)], dim=0)
            u_eff = torch.cat([qs_dequantize_if4(*blocks[("up", e)], N=LANE, K=H,
                                                 block_size=UE_VECTOR_SIZE)
                               for e in range(num_engines)], dim=0)
            d_eff = torch.cat([qs_dequantize_if4(*blocks[("down", e)], N=H, K=LANE,
                                                 block_size=UE_VECTOR_SIZE)
                               for e in range(num_engines)], dim=1)
        else:
            g_eff = qs_dequantize_if4(*blocks["gate"], N=MLP, K=H, block_size=UE_VECTOR_SIZE)
            u_eff = qs_dequantize_if4(*blocks["up"], N=MLP, K=H, block_size=UE_VECTOR_SIZE)
            d_eff = torch.cat([qs_dequantize_if4(*blocks[f"down{l}"], N=H, K=LANE,
                                                 block_size=UE_VECTOR_SIZE)
                               for l in range(lanes)], dim=1)
        a32 = a_host.float()
        gate_o = a32 @ g_eff.float().T
        silu = gate_o * torch.sigmoid(gate_o)
        ref = (silu * (a32 @ u_eff.float().T)) @ d_eff.float().T
        err = (got.float() - ref).abs().max().item()
        den = ref.abs().max().item() or 1.0
        res["snr_db"] = 20 * math.log10(den / err) if err > 0 else float("inf")
    return res


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("rowshard", "tp", "both"), default="both")
    p.add_argument("--engines", type=int, default=ENGINES)
    p.add_argument("--rows", type=int, default=ENGINES * 64,
                   help="global prefill rows M (default 512 = 8 x 64)")
    p.add_argument("--dev", default="xdma0")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0xAB1E)
    p.add_argument("--arena-base-gib", type=float, default=0.0,
                   help="start the arena this many GiB up, to separate a window "
                        "SIZE effect from an address RANGE effect")
    p.add_argument("--dual", action="store_true",
                   help="two 512 MB-aligned windows per core instead of one 1 GiB")
    p.add_argument("--window-mib", type=int, default=WINDOW_BYTES // MiB,
                   help="private window size; the DRAM controller is documented "
                        "to interleave across 512 MB windows")
    args = p.parse_args()

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    print(f"[window] {args.window_mib} MiB per core\n")
    assert MLP % args.engines == 0, f"MLP={MLP} does not split {args.engines} ways"
    assert (MLP // args.engines) % UE_VECTOR_SIZE == 0, "lane is not 64-aligned"

    modes = ("rowshard", "tp") if args.mode == "both" else (args.mode,)
    results = []
    for mode in modes:
        print(f"\n{'=' * 70}\n=== {mode} : M={args.rows} H={H} MLP={MLP} "
              f"on {args.engines} engines\n{'=' * 70}")
        results.append(run_case(mode, args.engines, args.rows, args.timeout,
                                args.seed, args.window_mib * MiB, args.dual,
                                int(args.arena_base_gib * 2**30)))

    print(f"\n{'=' * 70}\nONE MLP LAYER, {args.rows} prefill rows, "
          f"{args.engines} engines\n{'=' * 70}")
    print(f"{'mode':10s} {'latency us':>11s} {'GFLOPS':>9s} {'SNR dB':>8s} "
          f"{'priv MiB':>9s} {'shared MiB':>11s} {'ISA MiB':>8s}")
    for r in results:
        if r["hung"]:
            print(f"{r['mode']:10s} {'HUNG':>11s}")
            continue
        gf = r["flops"] / (r["latency_us"] * 1e-6) / 1e9
        print(f"{r['mode']:10s} {r['latency_us']:11.1f} {gf:9.1f} "
              f"{r['snr_db']:8.1f} {r['private_MiB']:9.2f} {r['shared_MiB']:11.2f} "
              f"{r['master_MiB']:8.3f}")
    if len(results) == 2 and not any(r["hung"] for r in results):
        base, tp = results[0], results[1]
        delta = 100.0 * (tp["latency_us"] - base["latency_us"]) / base["latency_us"]
        print(f"\ntensor-parallel is {abs(delta):.1f}% "
              f"{'SLOWER' if delta > 0 else 'FASTER'} than the row-shard baseline "
              f"on this layer")
        print(f"weight bytes per engine: rowshard {base['shared_MiB']:.1f} MiB shared "
              f"/ tp {tp['private_MiB']:.1f} MiB private")


if __name__ == "__main__":
    main()
