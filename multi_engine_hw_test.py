"""
Multi-engine hardware tests for the Unified Engine.

Split out of user_hw_test.py so the multi-engine work (M-row sharding,
N-column sharding, the cross-engine INSTRUCTION_FLAG rendezvous, and the
MultiEngineScheduler in multi_engine_shard.py) lives in its own file and does
not collide with upstream edits to the main suite.

Shared harness helpers (record_test, the RNG capture pair, write_test_summary)
are imported from user_hw_test, which is import-safe: its suite runs only under
`if __name__ == "__main__"`.

Usage:
    python multi_engine_hw_test.py [--dev xdma0] [--device kintex7]
"""

import argparse
import atexit
import math
import os
import sys
import time

import torch

from read_trace import generate_trace
import user_dma_core
from user_dma_core import (
    DMA_DEVICE_C2H,
    DMA_DEVICE_H2C,
    DMA_DEVICE_USER,
    TYPE,
    UE_VECTOR_SIZE,
    URAM_SECTION,
    calculate_snr,
    set_dma_device,
    UnifiedEngine,
)
from nn_lib import (
    eltwise_add_core_dram,
    eltwise_mul_core_dram,
    silu_core_dram,
)
from user_hw_test import (
    record_test,
    write_test_summary,
    _capture_rng_state,
    _restore_rng_state,
    _rng_aligned_randn_2d,
    _rng_state_fingerprint,
    precompute_freqs_cis,
)

class EngineShard:
    """One engine's slice of a row-sharded op chain: its UnifiedEngine, its
    row range within the full M, and its named DRAM tensors (each engine
    owns its own physical copy, never a shared address)."""

    def __init__(self, ue, row_offset: int, row_count: int):
        self.ue = ue
        self.row_offset = row_offset
        self.row_count = row_count
        self.tensors = {}
        self.prog_addr = None
class ShardGroup:
    """Bookkeeping for an N-engine, row-sharded (over M), barrier-free op
    chain. Replaces the hand-rolled parallel lists (a_addrs, b_addrs, ...)
    used in the first surgical N-engine tests -- every tensor is addressed
    by name per-shard instead of by matching list index across engines.

    Two upload/alloc patterns, matching the two roles a tensor can play:
      - sharded:   each engine gets its own row-count-sized region and its
                   own row-slice of the data (activations, per-row bias).
      - broadcast: each engine gets its own region of the SAME size holding
                   an IDENTICAL copy of the data (weights, K/V, gamma,
                   identity matrix) -- content is shared, DRAM address is
                   never shared (engines have separate physical DRAM/reg
                   files, see reference_instruction_flag_dual_engine_sync).

    engine0 (shards[0]) is the host-side sync point, same convention proven
    in the earlier surgical tests: workers run their chain then flag_set;
    engine0 runs its chain then checks every worker's flag before
    flag_set+halt. Host only ever waits on engine0's queue.
    """

    def __init__(self, num_engines: int, M: int,
                 engine_base_stride: int = 0x00010000, dram_base_stride: int = 0x10000000):
        import user_dma_core

        assert num_engines >= 2, f"num_engines must be >= 2, got {num_engines}"
        base, rem = divmod(M, num_engines)
        row_counts = [base + (1 if i < rem else 0) for i in range(num_engines)]
        row_offsets = [sum(row_counts[:i]) for i in range(num_engines)]
        assert all(m >= 1 for m in row_counts), f"num_engines={num_engines} too large for M={M}"

        self.M = M
        self.num_engines = num_engines
        self.shards = []
        for i in range(num_engines):
            ue = UnifiedEngine(
                BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * engine_base_stride,
                params_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride,
                tensor_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride + 0x08000000,
                program_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride + 0x0F000000,
            )
            self.shards.append(EngineShard(ue, row_offsets[i], row_counts[i]))

    def alloc_sharded(self, name: str, per_row_bytes: int) -> None:
        """Each shard's region is sized by ITS OWN row_count."""
        for shard in self.shards:
            shard.tensors[name] = shard.ue.allocate_tensor_dram(shard.row_count * per_row_bytes)

    def alloc_broadcast(self, name: str, total_bytes: int) -> None:
        """Every shard gets an identically-sized region for a full, shared copy."""
        for shard in self.shards:
            shard.tensors[name] = shard.ue.allocate_tensor_dram(total_bytes)

    def upload_sharded(self, name: str, full_tensor) -> None:
        """Slice full_tensor's rows per shard and upload each slice to that
        shard's own copy of `name`."""
        for shard in self.shards:
            rows = full_tensor[shard.row_offset: shard.row_offset + shard.row_count]
            shard.ue.dma_to_accelerator_memory(shard.tensors[name], rows)

    def upload_broadcast(self, name: str, tensor) -> None:
        """Upload an identical full copy of `tensor` to every shard's own
        copy of `name`."""
        for shard in self.shards:
            shard.ue.dma_to_accelerator_memory(shard.tensors[name], tensor)

    def compile_chain(self, chain_fn) -> None:
        """chain_fn(shard) emits this shard's op sequence via shard.ue.*_core(...)
        calls, reading/writing shard.tensors[...]. No barrier is emitted
        between ops inside chain_fn -- only around the whole chain, once per
        shard, following the proven engine0-waits-for-all-workers convention."""
        engine0 = self.shards[0]
        engine0.ue.start_capture()
        engine0.ue.generate_instruction_flag_clear()
        chain_fn(engine0)
        for i in range(1, self.num_engines):
            engine0.ue.generate_instruction_flag_check(target_engine_idx=i)
        engine0.ue.generate_instruction_flag_set()
        engine0.ue.generate_instruction_halt()
        engine0.ue.stop_capture()
        engine0.prog_addr = engine0.ue.get_program_dram_addr()
        engine0.ue.write_captured_instructions_to_dram(engine0.prog_addr)
        engine0.ue.allocate_program_dram(engine0.ue.get_capture_instruction_size_bytes())

        for shard in self.shards[1:]:
            shard.ue.start_capture()
            shard.ue.generate_instruction_flag_clear()
            chain_fn(shard)
            shard.ue.generate_instruction_flag_set()
            shard.ue.generate_instruction_halt()
            shard.ue.stop_capture()
            shard.prog_addr = shard.ue.get_program_dram_addr()
            shard.ue.write_captured_instructions_to_dram(shard.prog_addr)
            shard.ue.allocate_program_dram(shard.ue.get_capture_instruction_size_bytes())

    def launch_and_wait(self, wait_timeout_seconds: float = 10.0) -> None:
        engine0 = self.shards[0]
        for shard in self.shards[1:]:
            shard.ue.start_execute_from_dram(shard.prog_addr)
        engine0.ue.start_execute_from_dram(engine0.prog_addr)
        engine0.ue.wait_queue(wait_timeout_seconds)

    def download_sharded(self, name: str, cols: int):
        """Concatenate each shard's own copy of `name` (row_count x cols)
        into one [M, cols] tensor, in row order."""
        parts = [shard.ue.dma_from_accelerator_memory(shard.tensors[name], (shard.row_count, cols))
                 for shard in self.shards]
        return torch.cat(parts, dim=0)

    def trace(self, prefix: str) -> None:
        for i, shard in enumerate(self.shards):
            generate_trace(shard.ue, f"{prefix}_{self.num_engines}_{i}_{shard.row_count}.csv")

    def reset(self) -> None:
        for shard in self.shards:
            shard.ue.reset_tensor_dram_addr()
            shard.ue.clear_capture_buffer()
def flag_rendezvous_repeat_test(rounds: int = 27, chunk_elems: int = 1024,
                                filler_M: int = 128, filler_K: int = 512, filler_N: int = 512,
                                wait_timeout_seconds: float = 60.0, barrier: bool = True,
                                expect_pass: bool = True):
    """
    Does the cross-engine INSTRUCTION_FLAG rendezvous RE-ARM, many times, inside
    ONE captured instruction stream?

    Everything proven so far (ShardGroup.compile_chain, matmat_mul_two_engine_flag_check_test)
    fires set/check exactly once per program. This test performs ``rounds`` mutual
    rendezvous inside a single program per engine, with real cross-engine data
    exchange between them, so a barrier that fails to re-arm shows up as either a
    wait_queue timeout (hang on a stale CLEAR) or a wrong payload (false pass on a
    stale SET) -- never a silent pass.

    Data flow (DRAM is one flat shared space; the per-engine BASE_ADDR only selects
    AXI-Lite control registers, so engine1 can read an address engine0 wrote):

      round r, engine e (partner p):
        [filler matmul, only when r % 2 == e]   <- makes the partner genuinely WAIT
        src_e[r]      -> SRAM -> shared_e[r]    <- produce
        FLAG_SET; FLAG_CHECK(p)                 <- the rendezvous under test
        shared_p[r]   -> SRAM                   <- consume the OTHER engine's region
        FLAG_CLEAR                              <- re-arm for round r+1
        SRAM          -> dst_e[r]

    Verification is exact (pure bf16 copies): dst_e[r] must bit-match src_p[r] for
    every round. All shared/dst regions are pre-filled with a poison pattern, so a
    read that races ahead of the partner's write lands on poison and fails loudly.

    Clear placement matters. FLAG_CHECK only spin-waits on level-1, and an engine can
    only touch its OWN flag, so a two-engine repeating barrier has two races:
      * CLEAR too early -> partner misses our SET -> hang.
      * CLEAR too late  -> partner's NEXT round sees our stale SET -> reads poison.
    Putting CLEAR between the consume-read and the consume-write buys ~1 DMA of
    margin against the first (partner only has to retire one already-satisfied CHECK)
    and >=3 DMAs of margin against the second (partner cannot reach its next SET
    without a produce pair), so neither window is a few-cycle coin flip.
    """
    import user_dma_core

    num_engines = 2
    engine_base_stride = 0x00010000
    dram_base_stride = 0x10000000
    chunk_bytes = chunk_elems * 2
    sram_addr = 0x00000

    ues = []
    for i in range(num_engines):
        ues.append(UnifiedEngine(
            BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * engine_base_stride,
            params_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride,
            tensor_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride + 0x08000000,
            program_dram_base=user_dma_core.DRAM_START_ADDR + i * dram_base_stride + 0x0F000000,
        ))

    # Per-engine regions. Disjoint by construction (bases strided by 0x10000000)
    # even though the two allocator objects are independent.
    src_addrs, shared_addrs, dst_addrs = [], [], []
    fa_addrs, fb_addrs, fo_addrs = [], [], []
    for ue in ues:
        src_addrs.append(ue.allocate_tensor_dram(rounds * chunk_bytes))
        shared_addrs.append(ue.allocate_tensor_dram(rounds * chunk_bytes))
        dst_addrs.append(ue.allocate_tensor_dram(rounds * chunk_bytes))
        fa_addrs.append(ue.allocate_tensor_dram(filler_M * filler_K * 2))
        fb_addrs.append(ue.allocate_tensor_dram(filler_N * filler_K * 2))
        fo_addrs.append(ue.allocate_tensor_dram(filler_M * filler_N * 2))

    # --- Host-visible payloads -------------------------------------------------
    src = [torch.randn(rounds, chunk_elems, dtype=torch.bfloat16) for _ in range(num_engines)]
    poison = torch.full((rounds, chunk_elems), 7777.0, dtype=torch.bfloat16)
    filler_a = torch.randn(filler_M, filler_K, dtype=torch.bfloat16) / math.sqrt(filler_K)
    filler_b = torch.randn(filler_N, filler_K, dtype=torch.bfloat16)
    for i, ue in enumerate(ues):
        ue.dma_to_accelerator_memory(src_addrs[i], src[i])
        ue.dma_to_accelerator_memory(shared_addrs[i], poison)
        ue.dma_to_accelerator_memory(dst_addrs[i], poison)
        ue.dma_to_accelerator_memory(fa_addrs[i], filler_a)
        ue.dma_to_accelerator_memory(fb_addrs[i], filler_b)

    # --- Pre-clear both flags with a tiny program so a stale flag left set by an
    # earlier test cannot make round 0's CHECK pass spuriously. ----------------
    preclear_addrs = []
    for ue in ues:
        ue.start_capture()
        ue.generate_instruction_flag_clear()
        ue.generate_instruction_halt()
        ue.stop_capture()
        addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(addr)
        ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
        preclear_addrs.append(addr)
        ue.clear_capture_buffer()
    for i, ue in enumerate(ues):
        ue.start_execute_from_dram(preclear_addrs[i])
        ue.wait_queue(5.0)

    # --- The single captured multi-rendezvous program, one per engine ---------
    prog_addrs = []
    inst_bytes = []
    for e, ue in enumerate(ues):
        p = 1 - e
        ue.start_capture()
        ue.generate_instruction_flag_clear()
        for r in range(rounds):
            if (r % 2) == e:
                # Only one engine per round carries this, so the other engine has
                # to actually block at the rendezvous instead of coasting through.
                ue.matmat_mul_core(M=filler_M, K=filler_K, N=filler_N,
                                   A_DRAM_ADDR=fa_addrs[e], B_DRAM_ADDR=fb_addrs[e],
                                   OUTPUT_DRAM_ADDR=fo_addrs[e])
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=src_addrs[e] + r * chunk_bytes,
                sram_address=sram_addr, element_size=chunk_elems)
            ue.sram_to_accelerator_memory(
                sram_address=sram_addr,
                accelerator_dram_address=shared_addrs[e] + r * chunk_bytes,
                element_size=chunk_elems)
            if barrier:
                ue.generate_instruction_flag_set()
                ue.generate_instruction_flag_check(target_engine_idx=p)
            ue.accelerator_memory_to_sram(
                accelerator_dram_address=shared_addrs[p] + r * chunk_bytes,
                sram_address=sram_addr, element_size=chunk_elems)
            if barrier:
                ue.generate_instruction_flag_clear()
            ue.sram_to_accelerator_memory(
                sram_address=sram_addr,
                accelerator_dram_address=dst_addrs[e] + r * chunk_bytes,
                element_size=chunk_elems)
        ue.generate_instruction_halt()
        ue.stop_capture()
        addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(addr)
        inst_bytes.append(ue.get_capture_instruction_size_bytes())
        ue.allocate_program_dram(inst_bytes[-1])
        prog_addrs.append(addr)

    print(f"flag_rendezvous_repeat: rounds={rounds}, program bytes per engine={inst_bytes}")

    start = time.monotonic()
    for i, ue in enumerate(ues):
        ue.start_execute_from_dram(prog_addrs[i])
    timed_out = []
    for i, ue in enumerate(ues):
        ue.wait_queue(wait_timeout_seconds)
        if ue.is_queue_busy():
            timed_out.append(i)
    elapsed = time.monotonic() - start
    print(f"flag_rendezvous_repeat: elapsed {elapsed:.3f} s, timed_out engines={timed_out}")

    for i, ue in enumerate(ues):
        generate_trace(ue, f"flag_rendezvous_repeat_trace_{i}_{rounds}.csv")

    # --- Verify: each engine's dst[r] must be exactly the PARTNER's src[r] ----
    got = [ue.dma_from_accelerator_memory(dst_addrs[i], (rounds, chunk_elems))
           for i, ue in enumerate(ues)]
    ref = [src[1 - i] for i in range(num_engines)]

    first_bad = None
    for r in range(rounds):
        for e in range(num_engines):
            if not torch.equal(got[e][r].float(), ref[e][r].float()):
                if first_bad is None:
                    first_bad = (r, e)
                break
        if first_bad is not None:
            break
    bad_rounds = sum(1 for r in range(rounds)
                     for e in range(num_engines)
                     if not torch.equal(got[e][r].float(), ref[e][r].float()))

    snr = calculate_snr(torch.cat(ref, dim=0), torch.cat(got, dim=0))
    if first_bad is None:
        print(f"flag_rendezvous_repeat: all {rounds} rendezvous rounds exact on both engines "
              f"(SNR {snr:.2f} dB)")
    else:
        print(f"flag_rendezvous_repeat: FIRST MISMATCH at round {first_bad[0]} on engine{first_bad[1]}; "
              f"{bad_rounds}/{rounds * num_engines} (round, engine) payloads wrong; SNR {snr:.2f} dB")

    record_test("flag_rendezvous_repeat",
                f"rounds={rounds}, chunk_elems={chunk_elems}, num_engines={num_engines}",
                snr_db=snr, inst_bytes=max(inst_bytes))

    if not expect_pass:
        # Negative control: with the rendezvous removed the fast engine must read
        # poison. If this "passes", the test cannot detect a broken barrier.
        assert timed_out or first_bad is not None, \
            "barrier=False still produced correct data -- this test would not detect a broken rendezvous"
        print("flag_rendezvous_repeat: negative control OK (no-barrier run is detectably wrong)")
        for ue in ues:
            ue.reset_tensor_dram_addr()
            ue.clear_capture_buffer()
        return
    assert not timed_out, \
        (f"engines {timed_out} still busy after {wait_timeout_seconds:g}s -- the flag "
         f"rendezvous failed to re-arm (hang), first bad round={first_bad}")
    assert first_bad is None, \
        (f"cross-engine payload wrong from round {first_bad[0]} (engine{first_bad[1]}): the "
         f"rendezvous stopped holding after re-arm ({bad_rounds} bad payloads of "
         f"{rounds * num_engines})")

    for ue in ues:
        ue.reset_tensor_dram_addr()
        ue.clear_capture_buffer()
def matmat_mul_n_engine_test(M: int, K: int, N: int, num_engines: int = 4,
                              dynamic: bool = False, snr_threshold_db: float = 40.0):
    """
    Shard A along M across num_engines engines, each running the same
    matmat_mul_core against a shared B, synchronized via per-engine flag
    registers. Generalizes matmat_mul_two_cores (user_dma_core.py, 2-engine)
    and the flag-loop scaffolding from matmat_mul_multi_engine_flag_check_test
    to arbitrary N. Built on ShardGroup (see above) instead of hand-rolled
    parallel per-engine lists.
    """
    bytes_per_element = 2
    group = ShardGroup(num_engines, M)
    group.alloc_sharded('A', K * bytes_per_element)
    group.alloc_broadcast('B', N * K * bytes_per_element)
    group.alloc_sharded('OUT', N * bytes_per_element)

    def chain_fn(shard):
        ue = shard.ue
        m = shard.row_count
        m_reg = k_reg = n_reg = None
        if dynamic:
            m_reg = ue.alloc_isa_reg()
            k_reg = ue.alloc_isa_reg()
            n_reg = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(m_reg, m)
            ue.generate_instruction_add_set(k_reg, K)
            ue.generate_instruction_add_set(n_reg, N)
        ue.matmat_mul_core(
            M=m, K=K, N=N,
            A_DRAM_ADDR=shard.tensors['A'], B_DRAM_ADDR=shard.tensors['B'], OUTPUT_DRAM_ADDR=shard.tensors['OUT'],
            gpr_M_reg=m_reg, gpr_K_reg=k_reg, gpr_N_reg=n_reg,
        )
        if dynamic:
            ue.release_isa_reg(); ue.release_isa_reg(); ue.release_isa_reg()

    group.compile_chain(chain_fn)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    b = torch.randn(N, K, dtype=torch.bfloat16)
    group.upload_sharded('A', a)
    group.upload_broadcast('B', b)

    group.launch_and_wait(10.0)
    group.trace("matmat_mul_n_engine_trace")

    out_combined = group.download_sharded('OUT', N)
    ref = a @ b.T

    snr_combined = calculate_snr(ref, out_combined)
    print(f"N-engine ({num_engines}) sharded matmul SNR combined: {snr_combined:.2f} dB")
    for i, shard in enumerate(group.shards):
        ref_i = ref[shard.row_offset:shard.row_offset + shard.row_count, :]
        out_i = shard.ue.dma_from_accelerator_memory(shard.tensors['OUT'], (shard.row_count, N))
        snr_i = calculate_snr(ref_i, out_i)
        print(f"  engine{i} shard (rows {shard.row_offset}:{shard.row_offset + shard.row_count}) SNR: {snr_i:.2f} dB")

    assert snr_combined >= snr_threshold_db or snr_combined == float("inf"), \
        f"SNR {snr_combined:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("matmat_mul_n_engine",
                f"M={M}, K={K}, N={N}, num_engines={num_engines}, dynamic={dynamic}",
                snr_db=snr_combined)

    group.reset()
def matmat_mul_norm_chain_n_engine_test(M: int, K1: int, N1: int, N2: int,
                                         num_engines: int = 2, dynamic: bool = False,
                                         snr_threshold_db: float = 40.0):
    """
    Chained, sharded, barrier-free op sequence: matmul1 -> rms_norm -> matmul2,
    all row-sharded across num_engines engines with a SINGLE barrier at the end
    (no barrier between the three ops). Since matmul/norm are row-independent
    over M, each engine runs its whole per-shard chain with zero cross-engine
    sync, and only needs to rejoin (flag barrier) once, right before the host
    reads the full-width joined output. Built on ShardGroup (see above).
    """
    assert N1 % 64 == 0, f"N1 must be 64-aligned for the legacy rms_norm core, got {N1}"
    bytes_per_element = 2
    group = ShardGroup(num_engines, M)
    group.alloc_sharded('A', K1 * bytes_per_element)
    group.alloc_broadcast('B1', N1 * K1 * bytes_per_element)
    group.alloc_sharded('NORM_OUT', N1 * bytes_per_element)  # doubles as matmul1-out / norm-in-out scratch
    group.alloc_broadcast('GAMMA', N1 * bytes_per_element)
    group.alloc_broadcast('B2', N2 * N1 * bytes_per_element)
    group.alloc_sharded('OUT', N2 * bytes_per_element)

    def chain_fn(shard):
        ue = shard.ue
        m = shard.row_count
        # matmul1 writes straight into the rms_norm input region (NORM_OUT
        # doubles as scratch: matmul1 -> A, norm reads A writes NORM_OUT
        # in-place, matmul2 reads NORM_OUT writes final OUT). No barrier
        # between these three -- all row-independent over m.
        ue.matmat_mul_core(
            M=m, K=K1, N=N1,
            A_DRAM_ADDR=shard.tensors['A'], B_DRAM_ADDR=shard.tensors['B1'], OUTPUT_DRAM_ADDR=shard.tensors['NORM_OUT'],
        )
        ue.rms_norm_core_dram(
            M=m, N=N1, A_DRAM_ADDR=shard.tensors['NORM_OUT'], OUTPUT_DRAM_ADDR=shard.tensors['NORM_OUT'],
            GAMMA_DRAM_ADDR=shard.tensors['GAMMA'],
        )
        ue.matmat_mul_core(
            M=m, K=N1, N=N2,
            A_DRAM_ADDR=shard.tensors['NORM_OUT'], B_DRAM_ADDR=shard.tensors['B2'], OUTPUT_DRAM_ADDR=shard.tensors['OUT'],
        )

    group.compile_chain(chain_fn)

    # Host data: same B1/gamma/B2 broadcast to every engine (mirrors production
    # weight sharing), independent A shard per engine.
    a = torch.randn(M, K1, dtype=torch.bfloat16) / math.sqrt(K1)
    b1 = torch.randn(N1, K1, dtype=torch.bfloat16)
    gamma = torch.randn(N1, dtype=torch.bfloat16)
    b2 = torch.randn(N2, N1, dtype=torch.bfloat16)
    group.upload_sharded('A', a)
    group.upload_broadcast('B1', b1)
    group.upload_broadcast('GAMMA', gamma)
    group.upload_broadcast('B2', b2)

    group.launch_and_wait(10.0)
    group.trace("matmat_mul_norm_chain_n_engine_trace")

    out_combined = group.download_sharded('OUT', N2)

    rms = torch.nn.RMSNorm(N1)
    rms.weight.data = gamma
    ref_mid = rms((a @ b1.T))
    ref = ref_mid @ b2.T

    snr_combined = calculate_snr(ref, out_combined)
    print(f"N-engine ({num_engines}) sharded matmul->norm->matmul chain SNR combined: {snr_combined:.2f} dB")
    for i, shard in enumerate(group.shards):
        ref_i = ref[shard.row_offset:shard.row_offset + shard.row_count, :]
        out_i = shard.ue.dma_from_accelerator_memory(shard.tensors['OUT'], (shard.row_count, N2))
        snr_i = calculate_snr(ref_i, out_i)
        print(f"  engine{i} shard (rows {shard.row_offset}:{shard.row_offset + shard.row_count}) SNR: {snr_i:.2f} dB")

    assert snr_combined >= snr_threshold_db or snr_combined == float("inf"), \
        f"SNR {snr_combined:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("matmat_mul_norm_chain_n_engine",
                f"M={M}, K1={K1}, N1={N1}, N2={N2}, num_engines={num_engines}, dynamic={dynamic}",
                snr_db=snr_combined)

    group.reset()
def matmat_mul_norm_attn_chain_n_engine_test(M: int, K1: int, N1: int, aligned_seq_len: int, N2: int,
                                              num_engines: int = 2, snr_threshold_db: float = 30.0):
    """
    Chained, sharded, barrier-free op sequence: matmul1 -> rms_norm -> attention -> matmul2.

    IMPORTANT SCOPE NOTE: this shards the QUERY/batch dimension (M) across
    engines, same as matmat_mul_norm_chain_n_engine_test. Each engine holds a
    FULL, identical copy of K/V (broadcast, like a weight matrix) so attention
    over its Q shard needs no data from the other engine -- still a
    row-independent op, so still zero barriers mid-chain. This does NOT test
    the harder case of splitting the K/V sequence itself across engines
    (sequence-parallel attention), which would require an online-softmax
    partial-result merge at a real join point -- that is the next test to
    build if this one passes clean, since it is the case that actually needs
    a mid-chain barrier. Built on ShardGroup (see above).
    """
    assert N1 % 64 == 0, f"N1 must be 64-aligned for the legacy rms_norm core, got {N1}"
    assert aligned_seq_len % UE_VECTOR_SIZE == 0, \
        f"aligned_seq_len={aligned_seq_len} must be a multiple of {UE_VECTOR_SIZE}"
    head_dim = N1  # rms_norm output feeds directly into attention as Q
    bytes_per_element = 2

    group = ShardGroup(num_engines, M)
    group.alloc_sharded('A', K1 * bytes_per_element)
    group.alloc_broadcast('B1', N1 * K1 * bytes_per_element)
    group.alloc_sharded('NORM_OUT', N1 * bytes_per_element)
    group.alloc_broadcast('GAMMA', N1 * bytes_per_element)
    group.alloc_broadcast('K', aligned_seq_len * head_dim * bytes_per_element)
    group.alloc_broadcast('V', aligned_seq_len * head_dim * bytes_per_element)
    group.alloc_sharded('BIAS', aligned_seq_len * bytes_per_element)
    group.alloc_sharded('ATTN_OUT', head_dim * bytes_per_element)
    group.alloc_broadcast('IDENTITY', UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)
    group.alloc_broadcast('B2', N2 * head_dim * bytes_per_element)
    group.alloc_sharded('OUT', N2 * bytes_per_element)
    # SCRATCH has a per-shard-constant part (K/V-sized) plus a per-row part
    # (head_dim), so it doesn't fit the pure alloc_sharded/alloc_broadcast
    # shape -- allocate it directly per shard.
    for shard in group.shards:
        m = shard.row_count
        shard.tensors['SCRATCH'] = shard.ue.allocate_tensor_dram(
            (head_dim * aligned_seq_len + aligned_seq_len * aligned_seq_len + m * head_dim) * bytes_per_element
        )

    def chain_fn(shard):
        ue = shard.ue
        m = shard.row_count
        ue.matmat_mul_core(
            M=m, K=K1, N=N1,
            A_DRAM_ADDR=shard.tensors['A'], B_DRAM_ADDR=shard.tensors['B1'], OUTPUT_DRAM_ADDR=shard.tensors['NORM_OUT'],
        )
        ue.rms_norm_core_dram(
            M=m, N=N1, A_DRAM_ADDR=shard.tensors['NORM_OUT'], OUTPUT_DRAM_ADDR=shard.tensors['NORM_OUT'],
            GAMMA_DRAM_ADDR=shard.tensors['GAMMA'],
        )
        ue.unified_attention_core(
            batch=m, aligned_seq_len=aligned_seq_len, head_dim=head_dim,
            Q_DRAM_ADDR=shard.tensors['NORM_OUT'], K_DRAM_ADDR=shard.tensors['K'], V_DRAM_ADDR=shard.tensors['V'],
            BIAS_DRAM_ADDR=shard.tensors['BIAS'], OUTPUT_DRAM_ADDR=shard.tensors['ATTN_OUT'],
            SCRATCH_DRAM_ADDR=shard.tensors['SCRATCH'], IDENTITY_DRAM_ADDR=shard.tensors['IDENTITY'],
        )
        ue.matmat_mul_core(
            M=m, K=head_dim, N=N2,
            A_DRAM_ADDR=shard.tensors['ATTN_OUT'], B_DRAM_ADDR=shard.tensors['B2'], OUTPUT_DRAM_ADDR=shard.tensors['OUT'],
        )

    group.compile_chain(chain_fn)

    # Host data: B1/gamma/K/V/B2/identity broadcast identically to every engine
    # (mirrors production weight + KV-cache sharing), independent A/bias shard per engine.
    a = torch.randn(M, K1, dtype=torch.bfloat16) / math.sqrt(K1)
    b1 = torch.randn(N1, K1, dtype=torch.bfloat16)
    gamma = torch.randn(N1, dtype=torch.bfloat16)
    k = torch.randn(aligned_seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(aligned_seq_len, head_dim, dtype=torch.bfloat16)
    bias = torch.randn(M, aligned_seq_len, dtype=torch.bfloat16)
    identity = torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16)
    b2 = torch.randn(N2, head_dim, dtype=torch.bfloat16)

    group.upload_sharded('A', a)
    group.upload_broadcast('B1', b1)
    group.upload_broadcast('GAMMA', gamma)
    group.upload_broadcast('K', k)
    group.upload_broadcast('V', v)
    group.upload_sharded('BIAS', bias)
    group.upload_broadcast('IDENTITY', identity)
    group.upload_broadcast('B2', b2)

    group.launch_and_wait(50.0)
    group.trace("matmat_mul_norm_attn_chain_n_engine_trace")

    out_combined = group.download_sharded('OUT', N2)

    rms = torch.nn.RMSNorm(N1)
    rms.weight.data = gamma
    q = rms(a @ b1.T)
    q_scaled = q * (1.0 / math.sqrt(head_dim))
    scores = q_scaled @ k.t() + bias
    probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
    attn_out = probs @ v
    ref = attn_out @ b2.T

    snr_combined = calculate_snr(ref, out_combined)
    print(f"N-engine ({num_engines}) sharded matmul->norm->attention->matmul chain SNR combined: {snr_combined:.2f} dB")
    for i, shard in enumerate(group.shards):
        ref_i = ref[shard.row_offset:shard.row_offset + shard.row_count, :]
        out_i = shard.ue.dma_from_accelerator_memory(shard.tensors['OUT'], (shard.row_count, N2))
        snr_i = calculate_snr(ref_i, out_i)
        print(f"  engine{i} shard (rows {shard.row_offset}:{shard.row_offset + shard.row_count}) SNR: {snr_i:.2f} dB")

    assert snr_combined >= snr_threshold_db or snr_combined == float("inf"), \
        f"SNR {snr_combined:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("matmat_mul_norm_attn_chain_n_engine",
                f"M={M}, K1={K1}, N1={N1}, aligned_seq_len={aligned_seq_len}, N2={N2}, num_engines={num_engines}",
                snr_db=snr_combined)

    group.reset()
def _scheduler_chain_body_factory(K1, N1, N2, addrs, bpe=2):
    """Return a sharded-region body emitting matmul -> layer_norm -> matmul.

    Shared by the hardware test and the num_engines==1 byte-identity test so
    both legs emit provably the same op sequence.
    """
    def body(ctx):
        ctx.ue.matmat_mul_core(
            M=ctx.rows, K=K1, N=N1,
            A_DRAM_ADDR=ctx.rows_addr(addrs['A'], K1 * bpe),      # SHARED_ROWS
            B_DRAM_ADDR=addrs['B1'],                              # SHARED_FULL
            OUTPUT_DRAM_ADDR=ctx.rows_addr(addrs['MID'], N1 * bpe),
            gpr_M_reg=ctx.m_reg,
        )
        ctx.ue.layer_norm_core_dram(
            M=ctx.rows, N=N1,
            A_DRAM_ADDR=ctx.rows_addr(addrs['MID'], N1 * bpe),
            OUTPUT_DRAM_ADDR=ctx.rows_addr(addrs['MID'], N1 * bpe),
            GAMMA_DRAM_ADDR=addrs['GAMMA'], BETA_DRAM_ADDR=addrs['BETA'],
            ZEROS_DRAM_ADDR=ctx.per_engine('zeros'),              # PER_ENGINE
            gpr_M_reg=ctx.m_reg,
        )
        ctx.ue.matmat_mul_core(
            M=ctx.rows, K=N1, N=N2,
            A_DRAM_ADDR=ctx.rows_addr(addrs['MID'], N1 * bpe),
            B_DRAM_ADDR=addrs['B2'],
            OUTPUT_DRAM_ADDR=ctx.rows_addr(addrs['OUT'], N2 * bpe),
            gpr_M_reg=ctx.m_reg,
        )
    return body
def _scheduler_ref(a, b1, gamma, beta, b2, N1):
    mid = a @ b1.T
    ln = torch.nn.LayerNorm(N1, dtype=torch.float32)
    ln.weight.data = gamma.float()
    ln.bias.data = beta.float()
    return ln(mid.float()).to(torch.bfloat16) @ b2.T
def sharded_scheduler_chain_test(M: int = 256, K1: int = 2048, N1: int = 1024, N2: int = 512,
                                 num_engines: int = 2, snr_threshold_db: float = 40.0):
    """MultiEngineScheduler end-to-end: one SHARED full-size buffer set, row-offset
    addressing, PBI runtime row counts primed per engine, single end barrier.

    Unlike ShardGroup (which gives every engine its own physical copy of every
    tensor), here A/B1/GAMMA/BETA/B2/MID/OUT are allocated ONCE at the primary's
    addresses and both engines address the same flat DRAM -- exactly the shape a
    real model's compile function already has.
    """
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    primary = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                            params_dram_base=dram,
                            tensor_dram_base=dram + 0x08000000,
                            program_dram_base=dram + 0x0F000000)

    # One full-size allocation per tensor, at the "model's" addresses.
    addrs = {
        'A':     primary.allocate_tensor_dram(M * K1 * bpe),
        'B1':    primary.allocate_tensor_dram(N1 * K1 * bpe),
        'MID':   primary.allocate_tensor_dram(M * N1 * bpe),
        'GAMMA': primary.allocate_tensor_dram(N1 * bpe),
        'BETA':  primary.allocate_tensor_dram(N1 * bpe),
        'B2':    primary.allocate_tensor_dram(N2 * N1 * bpe),
        'OUT':   primary.allocate_tensor_dram(M * N2 * bpe),
    }
    zeros_primary = primary.allocate_tensor_dram(N1 * bpe)

    a = torch.randn(M, K1, dtype=torch.bfloat16) / math.sqrt(K1)
    b1 = torch.randn(N1, K1, dtype=torch.bfloat16)
    gamma = torch.randn(N1, dtype=torch.bfloat16)
    beta = torch.randn(N1, dtype=torch.bfloat16)
    b2 = torch.randn(N2, N1, dtype=torch.bfloat16)
    zeros = torch.zeros(N1, dtype=torch.bfloat16)

    primary.dma_to_accelerator_memory(addrs['A'], a)
    primary.dma_to_accelerator_memory(addrs['B1'], b1)
    primary.dma_to_accelerator_memory(addrs['GAMMA'], gamma)
    primary.dma_to_accelerator_memory(addrs['BETA'], beta)
    primary.dma_to_accelerator_memory(addrs['B2'], b2)
    primary.dma_to_accelerator_memory(zeros_primary, zeros)

    sched = MultiEngineScheduler(primary, num_engines=num_engines)
    sched.register_per_engine('zeros', zeros_primary, N1 * bpe, init_tensor=zeros)
    sched.preclear_flags()

    # --- compile: the primary owns its capture; the scheduler only emits into it ---
    primary.start_capture()
    primary.reset_isa_reg_counter()
    primary.reset_inst_ptr_counter()
    sched.begin_program()
    sched.sharded_region(M, _scheduler_chain_body_factory(K1, N1, N2, addrs, bpe))
    sched.finalize()
    primary.generate_instruction_halt()
    primary.stop_capture()
    prog_addr = primary.get_program_dram_addr()
    primary.write_captured_instructions_to_dram(prog_addr)
    primary.allocate_program_dram(primary.get_capture_instruction_size_bytes())
    inst_bytes = primary.get_capture_instruction_size_bytes() + sched.worker_program_bytes()

    sched.start_workers()
    primary.start_execute_from_dram(prog_addr)
    primary.wait_queue(30.0)

    out = primary.dma_from_accelerator_memory(addrs['OUT'], (M, N2))
    ref = _scheduler_ref(a, b1, gamma, beta, b2, N1)
    snr = calculate_snr(ref, out)
    print(f"MultiEngineScheduler sharded matmul->layer_norm->matmul "
          f"(num_engines={num_engines}, M={M}) SNR: {snr:.2f} dB")
    for (off, cnt), idx in zip(sched.split_rows(M), range(num_engines)):
        snr_i = calculate_snr(ref[off:off + cnt], out[off:off + cnt])
        print(f"  engine{idx} rows {off}:{off + cnt} SNR: {snr_i:.2f} dB")
    assert snr >= snr_threshold_db or snr == float("inf"), \
        f"SNR {snr:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("sharded_scheduler_chain",
                f"M={M}, K1={K1}, N1={N1}, N2={N2}, num_engines={num_engines}",
                snr_db=snr, inst_bytes=inst_bytes)

    primary.reset_tensor_dram_addr()
    primary.clear_capture_buffer()
    return snr
def sharded_scheduler_multi_region_test(M: int = 256, K: int = 512, H: int = 128,
                                        num_engines: int = 2, snr_threshold_db: float = 30.0):
    """Interleave: sharded(matmul+layer_norm) -> SINGLE-ENGINE attention -> sharded(...)
    -> single-engine attention -> sharded(...), all in ONE program per engine.

    Attention mixes every row with every other row, so a worker that races past a
    rendezvous (stale-flag bug) reads rows the primary has not written yet and the
    SNR collapses -- this is the test that the barrier actually RE-ARMS, not just
    that the arithmetic is right.
    """
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    primary = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                            params_dram_base=dram,
                            tensor_dram_base=dram + 0x08000000,
                            program_dram_base=dram + 0x0F000000)
    A = {
        'A':     primary.allocate_tensor_dram(M * K * bpe),
        'W1':    primary.allocate_tensor_dram(H * K * bpe),
        'MID':   primary.allocate_tensor_dram(M * H * bpe),
        'GAMMA': primary.allocate_tensor_dram(H * bpe),
        'BETA':  primary.allocate_tensor_dram(H * bpe),
        'W2':    primary.allocate_tensor_dram(H * H * bpe),
        'ATTN':  primary.allocate_tensor_dram(M * H * bpe),
        'ATTN2': primary.allocate_tensor_dram(M * H * bpe),
        'OUT':   primary.allocate_tensor_dram(M * H * bpe),
        'OUT2':  primary.allocate_tensor_dram(M * H * bpe),
        'K':     primary.allocate_tensor_dram(M * H * bpe),
        'V':     primary.allocate_tensor_dram(M * H * bpe),
        'BIAS':  primary.allocate_tensor_dram(M * M * bpe),
        'IDENT': primary.allocate_tensor_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bpe),
    }
    scratch = primary.allocate_tensor_dram((H * M + M * M + M * H) * bpe)
    zeros_primary = primary.allocate_tensor_dram(H * bpe)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    w1 = torch.randn(H, K, dtype=torch.bfloat16)
    w2 = torch.randn(H, H, dtype=torch.bfloat16) / math.sqrt(H)
    gamma = torch.randn(H, dtype=torch.bfloat16)
    beta = torch.randn(H, dtype=torch.bfloat16)
    kt = torch.randn(M, H, dtype=torch.bfloat16)
    vt = torch.randn(M, H, dtype=torch.bfloat16)
    bias = torch.randn(M, M, dtype=torch.bfloat16)
    zeros = torch.zeros(H, dtype=torch.bfloat16)
    for name, t in (('A', a), ('W1', w1), ('W2', w2), ('GAMMA', gamma), ('BETA', beta),
                    ('K', kt), ('V', vt), ('BIAS', bias),
                    ('IDENT', torch.eye(UE_VECTOR_SIZE, dtype=torch.bfloat16))):
        primary.dma_to_accelerator_memory(A[name], t)
    primary.dma_to_accelerator_memory(zeros_primary, zeros)

    sched = MultiEngineScheduler(primary, num_engines=num_engines)
    sched.register_per_engine('zeros', zeros_primary, H * bpe, init_tensor=zeros)
    sched.preclear_flags()

    def proj_norm(src, weight, dst, Kdim):
        """Sharded body: matmul(src @ weight.T) -> dst, then layer_norm in place."""
        def body(ctx):
            ctx.ue.matmat_mul_core(
                M=ctx.rows, K=Kdim, N=H,
                A_DRAM_ADDR=ctx.rows_addr(A[src], Kdim * bpe),
                B_DRAM_ADDR=A[weight],
                OUTPUT_DRAM_ADDR=ctx.rows_addr(A[dst], H * bpe),
                gpr_M_reg=ctx.m_reg)
            ctx.ue.layer_norm_core_dram(
                M=ctx.rows, N=H,
                A_DRAM_ADDR=ctx.rows_addr(A[dst], H * bpe),
                OUTPUT_DRAM_ADDR=ctx.rows_addr(A[dst], H * bpe),
                GAMMA_DRAM_ADDR=A['GAMMA'], BETA_DRAM_ADDR=A['BETA'],
                ZEROS_DRAM_ADDR=ctx.per_engine('zeros'), gpr_M_reg=ctx.m_reg)
        return body

    def solo_attention(q_addr, out_addr):
        """SINGLE-ENGINE region: emitted on the primary only (requirement 6)."""
        primary.unified_attention_core(
            batch=M, aligned_seq_len=M, head_dim=H,
            Q_DRAM_ADDR=q_addr, K_DRAM_ADDR=A['K'], V_DRAM_ADDR=A['V'],
            BIAS_DRAM_ADDR=A['BIAS'], OUTPUT_DRAM_ADDR=out_addr,
            SCRATCH_DRAM_ADDR=scratch, IDENTITY_DRAM_ADDR=A['IDENT'])

    primary.start_capture()
    primary.reset_isa_reg_counter()
    primary.reset_inst_ptr_counter()
    sched.begin_program()
    sched.sharded_region(M, proj_norm('A', 'W1', 'MID', K))
    solo_attention(A['MID'], A['ATTN'])
    sched.sharded_region(M, proj_norm('ATTN', 'W2', 'OUT', H))
    solo_attention(A['OUT'], A['ATTN2'])
    sched.sharded_region(M, proj_norm('ATTN2', 'W2', 'OUT2', H))
    sched.finalize()
    primary.generate_instruction_halt()
    primary.stop_capture()
    prog_addr = primary.get_program_dram_addr()
    primary.write_captured_instructions_to_dram(prog_addr)
    primary.allocate_program_dram(primary.get_capture_instruction_size_bytes())
    inst_bytes = primary.get_capture_instruction_size_bytes() + sched.worker_program_bytes()

    sched.start_workers()
    primary.start_execute_from_dram(prog_addr)
    primary.wait_queue(60.0)
    for idx, w in enumerate(sched.workers):
        assert not w.is_queue_busy(), f"worker engine{idx + 1} hung at a rendezvous"

    out = primary.dma_from_accelerator_memory(A['OUT2'], (M, H))

    ln = torch.nn.LayerNorm(H, dtype=torch.float32)
    ln.weight.data = gamma.float()
    ln.bias.data = beta.float()

    def ref_attn(q):
        scores = (q.float() * (1.0 / math.sqrt(H))) @ kt.float().t() + bias.float()
        return (torch.softmax(scores, dim=-1).to(torch.bfloat16) @ vt).to(torch.bfloat16)

    mid = ln((a @ w1.T).float()).to(torch.bfloat16)
    o1 = ln((ref_attn(mid) @ w2.T).float()).to(torch.bfloat16)
    ref = ln((ref_attn(o1) @ w2.T).float()).to(torch.bfloat16)

    snr = calculate_snr(ref, out)
    print(f"MultiEngineScheduler 3 sharded regions + 2 single-engine attentions "
          f"(num_engines={num_engines}, M={M}) SNR: {snr:.2f} dB")
    assert snr >= snr_threshold_db or snr == float("inf"), \
        f"SNR {snr:.2f} dB must be at least {snr_threshold_db:g} dB"
    record_test("sharded_scheduler_multi_region",
                f"M={M}, K={K}, H={H}, regions=3, num_engines={num_engines}",
                snr_db=snr, inst_bytes=inst_bytes)
    primary.reset_tensor_dram_addr()
    primary.clear_capture_buffer()
    return snr
def sharded_scheduler_passthrough_identity_test(M: int = 256, K1: int = 512, N1: int = 256,
                                                N2: int = 128):
    """num_engines==1 must emit a BYTE-IDENTICAL stream to hand-written
    single-engine emission (requirement 4). Compile-only, no hardware run."""
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler, capture_digest

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    ue = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                       params_dram_base=dram,
                       tensor_dram_base=dram + 0x08000000,
                       program_dram_base=dram + 0x0F000000)
    addrs = {k: ue.allocate_tensor_dram(sz) for k, sz in (
        ('A', M * K1 * bpe), ('B1', N1 * K1 * bpe), ('MID', M * N1 * bpe),
        ('GAMMA', N1 * bpe), ('BETA', N1 * bpe), ('B2', N2 * N1 * bpe),
        ('OUT', M * N2 * bpe))}
    zeros_addr = ue.allocate_tensor_dram(N1 * bpe)
    params_after_alloc = ue.get_params_dram_addr()

    # --- leg 1: hand-written single-engine emission ---
    ue.start_capture()
    ue.reset_isa_reg_counter()
    ue.reset_inst_ptr_counter()
    m_reg = ue.alloc_isa_reg()
    ue.generate_instruction_add_set(m_reg, M)
    ue.matmat_mul_core(M=M, K=K1, N=N1, A_DRAM_ADDR=addrs['A'], B_DRAM_ADDR=addrs['B1'],
                       OUTPUT_DRAM_ADDR=addrs['MID'], gpr_M_reg=m_reg)
    ue.layer_norm_core_dram(M=M, N=N1, A_DRAM_ADDR=addrs['MID'], OUTPUT_DRAM_ADDR=addrs['MID'],
                            GAMMA_DRAM_ADDR=addrs['GAMMA'], BETA_DRAM_ADDR=addrs['BETA'],
                            ZEROS_DRAM_ADDR=zeros_addr, gpr_M_reg=m_reg)
    ue.matmat_mul_core(M=M, K=N1, N=N2, A_DRAM_ADDR=addrs['MID'], B_DRAM_ADDR=addrs['B2'],
                       OUTPUT_DRAM_ADDR=addrs['OUT'], gpr_M_reg=m_reg)
    ue.stop_capture()
    baseline_digest = capture_digest(ue)
    baseline_n = len(ue.capture_buffer)
    ue.clear_capture_buffer()

    # --- leg 2: same body through the scheduler with num_engines=1 ---
    # Reset the params cursor so the layer_norm auto-allocated INV_N vector
    # lands at the same address as in leg 1 (it is baked into the stream).
    ue._next_params_dram_addr = params_after_alloc
    ue.start_capture()
    ue.reset_isa_reg_counter()
    ue.reset_inst_ptr_counter()
    sched = MultiEngineScheduler(ue, num_engines=1)
    sched.register_per_engine('zeros', zeros_addr, N1 * bpe)
    sched.begin_program()
    sched.sharded_region(M, _scheduler_chain_body_factory(K1, N1, N2, addrs, bpe))
    sched.finalize()
    ue.stop_capture()
    sharded_digest = capture_digest(ue)

    assert baseline_digest == sharded_digest, (
        f"num_engines=1 passthrough is NOT byte-identical: "
        f"baseline {baseline_digest[:16]} ({baseline_n} inst) vs "
        f"scheduler {sharded_digest[:16]} ({len(ue.capture_buffer)} inst)"
    )
    print(f"MultiEngineScheduler num_engines=1 passthrough: byte-identical "
          f"({baseline_n} instructions, sha256 {baseline_digest[:16]})")

    # Op allowlist must reject a non-row-independent op inside a region.
    rejected = False
    try:
        sched2 = MultiEngineScheduler(ue, num_engines=1)
        ue.start_capture()
        sched2.begin_program()
        sched2.sharded_region(M, lambda ctx: ctx.ue.unified_attention_core_dynamic())
    except AssertionError as exc:
        rejected = "not allowed inside a sharded region" in str(exc)
    finally:
        if ue.is_capture_on:
            ue.stop_capture()
        ue.clear_capture_buffer()
    assert rejected, "sharded region must refuse unified_attention_core_dynamic"
    print("MultiEngineScheduler op allowlist: attention correctly refused inside a sharded region")

    record_test("sharded_scheduler_passthrough_identity",
                f"M={M}, K1={K1}, N1={N1}, N2={N2}, num_engines=1",
                snr_db=float("inf"))
    ue.reset_tensor_dram_addr()
def _col_shard_gather(sched, name: str, M: int, N: int):
    """Host-side reassembly of the per-engine contiguous [M, cols] outputs.

    Each engine writes a DENSE [M, cols] block (see alloc_col_output), so the
    full [M, N] result is a column-wise concatenation in engine order. Nothing
    on the device gathers it -- the point of the mode is that downstream
    elementwise work stays in its own lane.
    """
    parts = []
    for (off, cols), ue, addr in zip(sched.split_cols(N), sched.engines,
                                     sched.col_output_addrs(name)):
        parts.append(ue.dma_from_accelerator_memory(addr, (M, cols)).to(torch.bfloat16))
    return torch.cat(parts, dim=1)
def sharded_scheduler_col_matmul_test(M: int = 64, K: int = 1024, N: int = 4096,
                                      num_engines: int = 2, quantized: bool = False,
                                      bias_enable: bool = False,
                                      int_variant: bool = True,
                                      snr_threshold_db: float = 40.0,
                                      return_output: bool = False):
    """One N-sharded matmul: engine i gets B rows [n0, n0+Nc) and writes its own
    dense [M, Nc] block.

    ``quantized=True`` exercises the IF4 path, where the scale blob must be
    sliced consistently with B -- the single most likely place for a column
    shard to go silently wrong. ``bias_enable=True`` exercises broadcast_N bias
    slicing, the vis_pos_embed-shaped trap: a length-N vector that looks like a
    broadcast constant but is indexed by N.
    """
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    primary = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                            params_dram_base=dram,
                            tensor_dram_base=dram + 0x08000000,
                            program_dram_base=dram + 0x0F000000)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    if quantized:
        b = (torch.rand(N, K, dtype=torch.bfloat16) * 2 - 1)
        b_addr, scale_addr = primary.quantize_weight(weight=b, N=N, K=K,
                                                     data_type=TYPE.IF4,
                                                     int_variant=int_variant)
        b_ref = primary.quantize_weight_simulate(b, data_type=TYPE.IF4,
                                                 int_variant=int_variant)
        data_type = TYPE.IF4
    else:
        b = torch.randn(N, K, dtype=torch.bfloat16)
        b_addr = primary.allocate_tensor_dram(N * K * bpe)
        primary.dma_to_accelerator_memory(b_addr, b)
        scale_addr = None
        b_ref = b
        data_type = None

    a_addr = primary.allocate_tensor_dram(M * K * bpe)
    primary.dma_to_accelerator_memory(a_addr, a)

    bias = c_addr = None
    if bias_enable:
        bias = torch.randn(N, dtype=torch.bfloat16)
        c_addr = primary.allocate_tensor_dram(N * bpe)
        primary.dma_to_accelerator_memory(c_addr, bias)

    sched = MultiEngineScheduler(primary, num_engines=num_engines)
    sched.alloc_col_output('OUT', M, N)
    sched.preclear_flags()

    def body(ctx):
        ctx.ue.matmat_mul_core(
            M=M, K=K, N=ctx.cols,
            A_DRAM_ADDR=a_addr,                                  # full A, shared
            B_DRAM_ADDR=ctx.b_addr(b_addr, K, data_type),        # B row block
            OUTPUT_DRAM_ADDR=ctx.col_out('OUT'),                 # dense [M, cols]
            C_DRAM_ADDR=(ctx.bias_addr(c_addr) if bias_enable else None),
            bias_mode="broadcast_N",
            is_B_quantized=quantized, data_type=data_type,
            SCALE_DRAM_ADDR=(ctx.scale_addr(scale_addr, K) if quantized else None),
        )

    primary.start_capture()
    primary.reset_isa_reg_counter()
    primary.reset_inst_ptr_counter()
    sched.begin_program()
    sched.col_sharded_region(N, body)
    sched.finalize()
    primary.generate_instruction_halt()
    primary.stop_capture()
    prog_addr = primary.get_program_dram_addr()
    primary.write_captured_instructions_to_dram(prog_addr)
    primary.allocate_program_dram(primary.get_capture_instruction_size_bytes())
    inst_bytes = primary.get_capture_instruction_size_bytes() + sched.worker_program_bytes()

    sched.start_workers()
    primary.start_execute_from_dram(prog_addr)
    primary.wait_queue(60.0)

    out = _col_shard_gather(sched, 'OUT', M, N)
    ref = a.float() @ b_ref.float().T
    if bias_enable:
        ref = ref + bias.float()
    ref = ref.to(torch.bfloat16)
    snr = calculate_snr(ref, out)
    tag = f"{'IF4-quant' if quantized else 'bf16'}{'+bias' if bias_enable else ''}"
    print(f"MultiEngineScheduler N-SHARDED matmul {tag} "
          f"(M={M}, K={K}, N={N}, num_engines={num_engines}) SNR: {snr:.2f} dB")
    for (off, cols), idx in zip(sched.split_cols(N), range(num_engines)):
        snr_i = calculate_snr(ref[:, off:off + cols], out[:, off:off + cols])
        print(f"  engine{idx} cols {off}:{off + cols} SNR: {snr_i:.2f} dB")
    assert snr >= snr_threshold_db or snr == float("inf"), \
        f"SNR {snr:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("sharded_scheduler_col_matmul",
                f"M={M}, K={K}, N={N}, {tag}, num_engines={num_engines}",
                snr_db=snr, inst_bytes=inst_bytes)

    primary.reset_tensor_dram_addr()
    primary.clear_capture_buffer()
    return out.clone() if return_output else snr
def sharded_scheduler_col_matmul_bitexact_test(M: int = 64, K: int = 1024, N: int = 4096):
    """N-sharded (2 engines) vs single-engine on IDENTICAL data: bit-exact.

    Splitting output columns changes NOTHING about the arithmetic of any single
    output element -- the full K reduction still happens in one matvec pass. So
    the bar here is not 40 dB, it is EQUALITY; anything less means a shard is
    reading the wrong B rows / scales / bias.
    """
    rng_state = _capture_rng_state()
    out1 = sharded_scheduler_col_matmul_test(M=M, K=K, N=N, num_engines=1,
                                             return_output=True)
    _restore_rng_state(rng_state)
    out2 = sharded_scheduler_col_matmul_test(M=M, K=K, N=N, num_engines=2,
                                             return_output=True)
    exact = torch.equal(out1, out2)
    snr = calculate_snr(out1, out2)
    print(f"MultiEngineScheduler N-sharded 1-engine vs 2-engine (M={M}, K={K}, N={N}): "
          f"{'BIT-EXACT' if exact else f'NOT bit-exact, SNR {snr:.2f} dB'}")
    assert exact, (
        f"N-sharding must not perturb the arithmetic: 1-engine vs 2-engine "
        f"differ (SNR {snr:.2f} dB, "
        f"{int((out1 != out2).sum())} / {out1.numel()} elements)")
    record_test("sharded_scheduler_col_matmul_bitexact",
                f"M={M}, K={K}, N={N}, 1 vs 2 engines", snr_db=float("inf"))
def sharded_scheduler_col_mlp_chain_test(M: int = 64, K: int = 1024, N: int = 4096,
                                         num_engines: int = 2,
                                         snr_threshold_db: float = 40.0):
    """The denoise MLP lane: gate + up + gelu-multiply, N-sharded, NO JOIN.

    gate = gelu(x @ Wg^T), up = x @ Wu^T, h = gate * up. Every one of those is a
    per-column-block operation, so engine i touches columns [n0, n0+Nc) and
    NOTHING ELSE for the whole sequence -- three ops, one region, zero
    mid-chain barriers. This is the win the mode exists for; if a shard were
    reading outside its lane the elementwise multiply would pair mismatched
    columns and the SNR would collapse.
    """
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    primary = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                            params_dram_base=dram,
                            tensor_dram_base=dram + 0x08000000,
                            program_dram_base=dram + 0x0F000000)

    x = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    wg = torch.randn(N, K, dtype=torch.bfloat16) / math.sqrt(K)
    wu = torch.randn(N, K, dtype=torch.bfloat16) / math.sqrt(K)

    x_addr = primary.allocate_tensor_dram(M * K * bpe)
    wg_addr = primary.allocate_tensor_dram(N * K * bpe)
    wu_addr = primary.allocate_tensor_dram(N * K * bpe)
    primary.dma_to_accelerator_memory(x_addr, x)
    primary.dma_to_accelerator_memory(wg_addr, wg)
    primary.dma_to_accelerator_memory(wu_addr, wu)

    sched = MultiEngineScheduler(primary, num_engines=num_engines)
    sched.alloc_col_output('GATE', M, N)
    sched.alloc_col_output('UP', M, N)
    sched.alloc_col_output('H', M, N)
    sched.preclear_flags()

    def body(ctx):
        ctx.ue.matmat_mul_core(
            M=M, K=K, N=ctx.cols, A_DRAM_ADDR=x_addr,
            B_DRAM_ADDR=ctx.b_addr(wg_addr, K),
            OUTPUT_DRAM_ADDR=ctx.col_out('GATE'), gelu_enable=True)
        ctx.ue.matmat_mul_core(
            M=M, K=K, N=ctx.cols, A_DRAM_ADDR=x_addr,
            B_DRAM_ADDR=ctx.b_addr(wu_addr, K),
            OUTPUT_DRAM_ADDR=ctx.col_out('UP'))
        # Stays entirely in this engine's column lane: no barrier, no peer data.
        ctx.ue.eltwise_core_dram(
            M=M, N=ctx.cols, dram_a=ctx.col_out('GATE'), dram_b=ctx.col_out('UP'),
            dram_out=ctx.col_out('H'), mode=user_dma_core.UE_MODE.ELTWISE_MUL)

    primary.start_capture()
    primary.reset_isa_reg_counter()
    primary.reset_inst_ptr_counter()
    sched.begin_program()
    sched.col_sharded_region(N, body)
    sched.finalize()
    primary.generate_instruction_halt()
    primary.stop_capture()
    prog_addr = primary.get_program_dram_addr()
    primary.write_captured_instructions_to_dram(prog_addr)
    primary.allocate_program_dram(primary.get_capture_instruction_size_bytes())
    inst_bytes = primary.get_capture_instruction_size_bytes() + sched.worker_program_bytes()

    sched.start_workers()
    primary.start_execute_from_dram(prog_addr)
    primary.wait_queue(60.0)

    out = _col_shard_gather(sched, 'H', M, N)
    # The LALU's GELU is the sigmoid approximation x*sigmoid(1.702x), not the
    # tanh one -- matching the reference used by the eltwise/matmul activation
    # tests above.
    pre = (x.float() @ wg.float().T)
    gate = (pre * torch.sigmoid(1.702 * pre)).to(torch.bfloat16)
    up = (x.float() @ wu.float().T).to(torch.bfloat16)
    ref = (gate.float() * up.float()).to(torch.bfloat16)
    snr = calculate_snr(ref, out)
    print(f"MultiEngineScheduler N-SHARDED MLP lane gate/up/gelu-mul, NO JOIN "
          f"(M={M}, K={K}, N={N}, num_engines={num_engines}) SNR: {snr:.2f} dB")
    for (off, cols), idx in zip(sched.split_cols(N), range(num_engines)):
        print(f"  engine{idx} cols {off}:{off + cols} SNR: "
              f"{calculate_snr(ref[:, off:off + cols], out[:, off:off + cols]):.2f} dB")
    assert snr >= snr_threshold_db or snr == float("inf"), \
        f"SNR {snr:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("sharded_scheduler_col_mlp_chain",
                f"M={M}, K={K}, N={N}, num_engines={num_engines}, no join",
                snr_db=snr, inst_bytes=inst_bytes)

    primary.reset_tensor_dram_addr()
    primary.clear_capture_buffer()
    return snr
def sharded_scheduler_k_split_reduce_test(M: int = 64, K: int = 4096, N: int = 1024,
                                          num_engines: int = 2,
                                          stub_reduction: bool = False,
                                          snr_threshold_db: float = 40.0):
    """mlp down (4096 -> 1024): the split axis is K, so this is a REDUCTION.

    Each engine multiplies its K-slice and produces a FULL [M, N] PARTIAL SUM;
    ``reduce_add`` then adds them. B is N x K row-major, so a K-slice is a
    strided COLUMN slice of B's rows and CANNOT be reached by shifting
    B_DRAM_ADDR -- the sliced weights are prepared here on the host and
    uploaded as separate contiguous blobs, which in a model is one-time load
    prep.

    ``stub_reduction=True`` is the NEGATIVE CONTROL: skip the barrier + add and
    keep only engine 0's partial. The answer is then roughly half the true sum,
    and the assertion inverts -- a gate that cannot fail proves nothing.
    """
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    primary = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                            params_dram_base=dram,
                            tensor_dram_base=dram + 0x08000000,
                            program_dram_base=dram + 0x0F000000)

    a = torch.randn(M, K, dtype=torch.bfloat16) / math.sqrt(K)
    b = torch.randn(N, K, dtype=torch.bfloat16) / math.sqrt(K)

    sched = MultiEngineScheduler(primary, num_engines=num_engines)
    k_split = sched.split_k(K)

    # Host-side pre-slice: A[:, k0:k0+Kc] and B[:, k0:k0+Kc], each contiguous.
    a_addrs, b_addrs, partial_addrs = [], [], []
    for (k0, kc), ue in zip(k_split, sched.engines):
        aa = primary.allocate_tensor_dram(M * kc * bpe, align_bytes=128)
        bb = primary.allocate_tensor_dram(N * kc * bpe, align_bytes=128)
        pp = primary.allocate_tensor_dram(M * N * bpe, align_bytes=128)
        primary.dma_to_accelerator_memory(aa, a[:, k0:k0 + kc].contiguous())
        primary.dma_to_accelerator_memory(bb, b[:, k0:k0 + kc].contiguous())
        a_addrs.append(aa); b_addrs.append(bb); partial_addrs.append(pp)
    out_addr = primary.allocate_tensor_dram(M * N * bpe, align_bytes=128)
    primary.dma_to_accelerator_memory(out_addr, torch.zeros(M, N, dtype=torch.bfloat16))
    sched.preclear_flags()

    def body(ctx):
        i = ctx.engine_idx
        # Every engine emits the FULL [M, N] -- as a PARTIAL SUM over its K slice.
        ctx.ue.matmat_mul_core(
            M=M, K=ctx.k_cols, N=N,
            A_DRAM_ADDR=a_addrs[i], B_DRAM_ADDR=b_addrs[i],
            OUTPUT_DRAM_ADDR=partial_addrs[i])

    primary.start_capture()
    primary.reset_isa_reg_counter()
    primary.reset_inst_ptr_counter()
    sched.begin_program()
    sched.k_sharded_region(K, body)
    if stub_reduction:
        # NEGATIVE CONTROL: no barrier, no add. Keep engine 0's partial only.
        primary.eltwise_core_dram(M=M, N=N, dram_a=partial_addrs[0], dram_b=None,
                                  dram_out=out_addr,
                                  mode=user_dma_core.UE_MODE.MUL_BROADCAST, scalar=1.0)
    else:
        sched.reduce_add(partial_addrs, out_addr, M, N)
    sched.finalize()
    primary.generate_instruction_halt()
    primary.stop_capture()
    prog_addr = primary.get_program_dram_addr()
    primary.write_captured_instructions_to_dram(prog_addr)
    primary.allocate_program_dram(primary.get_capture_instruction_size_bytes())
    inst_bytes = primary.get_capture_instruction_size_bytes() + sched.worker_program_bytes()

    sched.start_workers()
    primary.start_execute_from_dram(prog_addr)
    primary.wait_queue(60.0)

    out = primary.dma_from_accelerator_memory(out_addr, (M, N))
    ref = (a.float() @ b.float().T).to(torch.bfloat16)
    snr = calculate_snr(ref, out)
    label = "STUBBED (negative control)" if stub_reduction else "reduce_add"
    print(f"MultiEngineScheduler K-SPLIT + {label} "
          f"(M={M}, K={K}, N={N}, num_engines={num_engines}) SNR: {snr:.2f} dB")
    if stub_reduction:
        assert snr < snr_threshold_db, (
            f"NEGATIVE CONTROL FAILED: stubbing out the cross-engine reduction still "
            f"gave {snr:.2f} dB (>= {snr_threshold_db:g} dB). The reduction test is "
            f"not actually testing the reduction.")
        print(f"  negative control OK: {snr:.2f} dB < {snr_threshold_db:g} dB "
              f"without the cross-engine add")
    else:
        assert snr >= snr_threshold_db or snr == float("inf"), \
            f"SNR {snr:.2f} dB must be at least {snr_threshold_db:g} dB"

    record_test("sharded_scheduler_k_split_reduce" + ("_negctl" if stub_reduction else ""),
                f"M={M}, K={K}, N={N}, num_engines={num_engines}, {label}",
                snr_db=snr, inst_bytes=inst_bytes)

    primary.reset_tensor_dram_addr()
    primary.clear_capture_buffer()
    return snr
def sharded_scheduler_col_passthrough_identity_test(M: int = 64, K: int = 1024, N: int = 4096):
    """num_engines==1 N-sharding must be an EXACT passthrough: byte-identical to
    hand-written single-engine emission, not one extra instruction.

    Also checks the alignment contract fails loudly: a column shard that is not
    a multiple of 64 elements must be refused at compile time, not rounded.
    Compile-only, no hardware run.
    """
    import user_dma_core
    from multi_engine_shard import MultiEngineScheduler, capture_digest

    bpe = 2
    dram = user_dma_core.DRAM_START_ADDR
    ue = UnifiedEngine(BASE_ADDR=user_dma_core.UE_0_BASE_ADDR,
                       params_dram_base=dram,
                       tensor_dram_base=dram + 0x08000000,
                       program_dram_base=dram + 0x0F000000)
    a_addr = ue.allocate_tensor_dram(M * K * bpe)
    wg_addr = ue.allocate_tensor_dram(N * K * bpe)
    c_addr = ue.allocate_tensor_dram(N * bpe)
    tensor_after_alloc = ue.get_tensor_dram_addr()
    params_after_alloc = ue.get_params_dram_addr()

    # --- leg 1: hand-written single-engine emission ---
    out_addr = ue.allocate_tensor_dram(M * N * bpe, align_bytes=128)
    ue.start_capture()
    ue.reset_isa_reg_counter()
    ue.reset_inst_ptr_counter()
    ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a_addr, B_DRAM_ADDR=wg_addr,
                       OUTPUT_DRAM_ADDR=out_addr, C_DRAM_ADDR=c_addr,
                       bias_mode="broadcast_N", gelu_enable=True)
    ue.stop_capture()
    baseline_digest = capture_digest(ue)
    baseline_n = len(ue.capture_buffer)
    ue.clear_capture_buffer()

    # --- leg 2: same body through the scheduler with num_engines=1 ---
    ue._tensor_dram_addr = tensor_after_alloc
    ue._next_params_dram_addr = params_after_alloc
    ue.start_capture()
    ue.reset_isa_reg_counter()
    ue.reset_inst_ptr_counter()
    sched = MultiEngineScheduler(ue, num_engines=1)
    sched.alloc_col_output('OUT', M, N)
    sched.begin_program()
    sched.col_sharded_region(N, lambda ctx: ctx.ue.matmat_mul_core(
        M=M, K=K, N=ctx.cols, A_DRAM_ADDR=a_addr,
        B_DRAM_ADDR=ctx.b_addr(wg_addr, K),
        OUTPUT_DRAM_ADDR=ctx.col_out('OUT'),
        C_DRAM_ADDR=ctx.bias_addr(c_addr), bias_mode="broadcast_N",
        gelu_enable=True))
    sched.finalize()
    ue.stop_capture()
    sharded_digest = capture_digest(ue)
    assert baseline_digest == sharded_digest, (
        f"num_engines=1 N-shard passthrough is NOT byte-identical: "
        f"baseline {baseline_digest[:16]} ({baseline_n} inst) vs "
        f"scheduler {sharded_digest[:16]} ({len(ue.capture_buffer)} inst)")
    print(f"MultiEngineScheduler num_engines=1 N-shard passthrough: byte-identical "
          f"({baseline_n} instructions, sha256 {baseline_digest[:16]})")
    ue.clear_capture_buffer()

    # Alignment contract: 64-element column granularity is a hard requirement.
    sched2 = MultiEngineScheduler(ue, num_engines=2)
    rejected = False
    try:
        sched2.split_cols(4000)   # 4000 / 64 = 62.5
    except AssertionError as exc:
        rejected = "not a multiple of col_align" in str(exc)
    assert rejected, "split_cols must refuse an N that is not 64-element aligned"
    # 128-byte SRAM-row contract on every computed address.
    from multi_engine_shard import _shifted
    row_rejected = False
    try:
        _shifted(0x80000000, 96, "deliberately bad shard")
    except AssertionError as exc:
        row_rejected = "SRAM row" in str(exc)
    assert row_rejected, "_shifted must refuse a non-128-byte-aligned shard offset"
    print("MultiEngineScheduler N-shard alignment gates: "
          "unaligned N and unaligned offset both correctly refused")

    record_test("sharded_scheduler_col_passthrough_identity",
                f"M={M}, K={K}, N={N}, num_engines=1", snr_db=float("inf"))
    ue.reset_tensor_dram_addr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-engine hardware tests")
    parser.add_argument('--dev', type=str, default='xdma0')
    parser.add_argument('--device', type=str, default='kintex7')
    parser.add_argument('--base-addr', type=lambda x: int(x, 0), default=None)
    args = parser.parse_args()

    set_dma_device("efinix" if args.device == "efinix" else args.dev,
                   base_addr=args.base_addr)
    axi_width_bits = 512 if args.device in ("bittware", "rk") else 256
    os.environ["UE_AXI_DATA_WIDTH_BITS"] = str(axi_width_bits)
    user_dma_core.UE_AXI_DATA_WIDTH_BITS = axi_width_bits
    globals()["UE_AXI_DATA_WIDTH_BITS"] = axi_width_bits

    torch.manual_seed(0)

    # --- Cross-engine rendezvous: does the flag re-arm across many barriers
    # inside ONE captured stream? (27 = pi05 vision encoder layer count.)
    flag_rendezvous_repeat_test(rounds=27)

    # --- M-axis (row) sharding ---
    matmat_mul_n_engine_test(M=256, K=2048, N=1024, num_engines=2, dynamic=False)
    matmat_mul_n_engine_test(M=256, K=2048, N=1024, num_engines=2, dynamic=True)
    matmat_mul_norm_chain_n_engine_test(M=256, K1=2048, N1=1024, N2=512, num_engines=2)
    matmat_mul_norm_attn_chain_n_engine_test(M=256, K1=2048, N1=128,
                                             aligned_seq_len=256, N2=512, num_engines=2)

    # --- MultiEngineScheduler, M-axis ---
    sharded_scheduler_passthrough_identity_test()
    sharded_scheduler_chain_test(num_engines=1)
    sharded_scheduler_chain_test(num_engines=2)
    sharded_scheduler_multi_region_test(num_engines=1)
    sharded_scheduler_multi_region_test(num_engines=2)

    # --- MultiEngineScheduler, N-axis (column) sharding ---
    sharded_scheduler_col_passthrough_identity_test()
    sharded_scheduler_col_matmul_bitexact_test(M=64, K=1024, N=4096)
    sharded_scheduler_col_matmul_test(M=64, K=1024, N=4096, num_engines=1)
    sharded_scheduler_col_matmul_test(M=64, K=1024, N=4096, num_engines=2)
    sharded_scheduler_col_matmul_test(M=64, K=1024, N=2048, num_engines=2,
                                      bias_enable=True)
    sharded_scheduler_col_matmul_test(M=64, K=1024, N=4096, num_engines=2,
                                      quantized=True)
    sharded_scheduler_col_matmul_test(M=64, K=1024, N=2048, num_engines=2,
                                      quantized=True, bias_enable=True)
    sharded_scheduler_col_mlp_chain_test(M=64, K=1024, N=4096, num_engines=2)
    sharded_scheduler_k_split_reduce_test(M=64, K=4096, N=1024, num_engines=2)
    # Negative control: stub the cross-engine reduction, the same test must FAIL.
    sharded_scheduler_k_split_reduce_test(M=64, K=4096, N=1024, num_engines=2,
                                          stub_reduction=True)

    write_test_summary("multi_engine_hw_test_summary.md")
    print("Status: ALL MULTI-ENGINE TESTS PASSED")
