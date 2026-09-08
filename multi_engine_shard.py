"""
Row-sharded (M-split) and column-sharded (N-split) multi-engine emission for
EXISTING single-engine model compile functions.

WHY A NEW MODULE (and not more of ``ShardGroup`` in user_hw_test.py):
``ShardGroup`` owns everything -- it constructs its own engines, allocates its
own per-engine tensor copies, opens/closes its own capture, and emits its own
barrier. That is right for a self-contained hardware test and wrong for a
model, whose buffers/addresses/capture already exist and whose program must
interleave sharded regions with single-engine regions. This module is a
library that models (``models/**/*_test.py``) will import, so it lives at the
repo root next to ``user_dma_core.py`` rather than inside the test runner.

MODEL OF EXECUTION
------------------
* DRAM is ONE FLAT SHARED ADDRESS SPACE (0x80000000..0xffffffff). A
  ``UnifiedEngine``'s ``BASE_ADDR`` only selects its AXI-Lite control
  registers; ``params/tensor/program_dram_base`` are just software allocator
  cursors. So weights are NOT duplicated, engines hand data to each other by
  address, and the ONLY thing that must be centrally owned is allocation --
  hence this class owns the worker engines' allocator bases.
* The model object itself (a ``UnifiedEngine`` subclass) is engine 0, "the
  primary". Its capture session is the model's own ``start_capture()``; this
  class never opens or closes it. Worker engines get their own capture,
  opened by :meth:`begin_program` and closed by :meth:`finalize`.
* A sharded region body is a CALLABLE, replayed once per engine (straight-line
  ``self.op(...)`` code cannot be "forked" into N instruction streams any other
  way). Every engine's stream stays one continuous program.

TENSOR ROLES (see :class:`TensorRole`)
--------------------------------------
SHARED_ROWS  one full-size buffer at its existing model address; each engine
             addresses its own row block via ``ctx.rows_addr(base, row_bytes)``.
SHARED_FULL  one buffer every engine reads in full (weights, K/V, pos tables,
             identity, zero-bias): pass the address through unchanged.
PER_ENGINE   scratch that is WRITTEN as scratch and therefore must be
             duplicated (pi05's ``vis_zeros_addr``, which layer_norm_core_dram
             writes into -- see models/pi05/pi05_test.py:2265).
             Register it with :meth:`register_per_engine`; the primary keeps
             its existing address, workers get scheduler-allocated copies.

KNOWN TRAPS THIS MODULE DOES NOT AND CANNOT ENFORCE
---------------------------------------------------
* Strided SRAM->DRAM copies need a PER-INDEX destination row offset. A missing
  one yields finite-but-scrambled output, not NaN. Such copies are part of the
  attention marshalling, which stays single-engine in v1.
* CONSTRUCTING A ``UnifiedEngine`` DESTROYS THE FIRST 16 KB OF DRAM:
  ``init_unified_engine()`` runs a DRAM self-test against the HARDCODED
  ``DRAM_START_ADDR``, ignoring ``params_dram_base``
  (``user_dma_core.py:827-837``). This class snapshots and restores that region
  around worker construction so its own callers are safe -- but ANY other code
  that builds an engine after uploading weights is not.
* ``ue_selector``: ``runtime_addr`` and ``append_row`` share one scratch GPR
  (``_addr_tmp``); emit every op that uses a computed address BEFORE any
  ``append_row``.

N-AXIS (OUTPUT-COLUMN) SHARDING -- the second mode
--------------------------------------------------
M-sharding is useless where M is a single 64-row block. The pi05 action expert
runs 18 layers x 10 Euler steps at M = 64 (``AE_ACTION_HORIZON_PADDED``) and
cannot be row-split at all, but its N dimensions are fat (MLP 1024->4096, q
1024->2048, o 2048->1024). N-sharding splits the OUTPUT COLUMNS instead.

It is cheap for exactly one reason, and the reason is a layout fact
(``user_dma_core.py:5375``)::

    A is M x K row-major.  B is N x K row-major; the accelerator applies an
    implicit transpose, so the result is A @ B^T, i.e. M x N.

Because ``B`` is stored **N x K**, output columns ``[n0, n0+Nc)`` are a
**CONTIGUOUS ROW BLOCK of B** at ``B + n0*K*bpe``. No strided weight gather, no
weight duplication -- engine i just passes a shifted ``B_DRAM_ADDR`` and
``N=Nc``. And ``matmat_mul_core`` already tiles N internally into ``N_chunk``s
sized to URAM_B (``user_dma_core.py:5423``), so a smaller N is a well-trodden
path through the same code.

OUTPUT IS NOT STRIDED -- IT IS PER-ENGINE CONTIGUOUS.  A column block of a full
[M, N] buffer would be strided (Nc elements every N), and the dynamic/PBI path
does not support strided writeback ("Strided writeback (opt 5) deferred",
``user_dma_core.py:5688``). We do not need it: the legacy writeback stride is
``N * bpe`` with ``N`` = **the N that was passed in** (``user_dma_core.py:5578``),
so calling the core with ``N=Nc`` writes a DENSE, CONTIGUOUS ``[M, Nc]`` block.
Each engine therefore owns its own ``[M, Nc]`` buffer -- option (a) -- which is
not a workaround but what the kernel does natively. :meth:`alloc_col_output`
allocates them (128-byte aligned) and ``ctx.col_out(name)`` resolves them.

STAY SHARDED ACROSS OPS -- the actual win.  A column shard of the output is a
column shard of every ELEMENTWISE thing downstream. The denoise MLP
(``gate = x@Wg^T``, ``up = x@Wu^T``, ``h = gelu(gate)*up``) splits to
``[64, 2048]`` per engine and the GELU-multiply never leaves its lane, so a
whole gate/up/activation sequence runs with ZERO barriers. Emit it as one
region body and pass ``join=False`` between regions that stay in the same lane.

THE FOUR PER-COLUMN OFFSETS (get any one wrong and you get finite garbage)::

    B      (bf16)  base + n0 * K * bpe
    B      (IF8)   base + n0 * K
    B      (IF4)   base + (n0 * K) // 2          # two nibbles per byte
    scale  (IF4/8) base + ((n0 * K) // UE_VECTOR_SIZE) * bpe
    bias   (broadcast_N)  base + n0 * bpe
    OUT                   a separate per-engine [M, Nc] buffer (see above)

The scale offset is the one to be careful with. ``quantize_weight`` writes ONE
bf16 scale per 64-element block of the FLATTENED (N, K) matrix, and the core
reads it as ``SCALE + ((j*K)//UE_VECTOR_SIZE)*bpe`` for column chunk ``j``
(``user_dma_core.py:5523``, ``:7519``). So the scale blob is effectively
``[N, K/64]`` bf16 and a column shard takes a contiguous row block of it --
the same shape of slice as B itself. This is why ``K % UE_VECTOR_SIZE == 0`` is
asserted: the scale stride is linear in whole K-blocks.

THE BIAS IS A vis_pos_embed TRAP.  ``bias_mode="broadcast_N"`` reads a length-N
vector and looks like a broadcast constant, so it is exactly the kind of thing
that gets passed through unsliced and silently corrupts N-1 of every N columns.
:meth:`ColumnShardContext.bias_addr` exists so no call site hand-rolls it.

K-SPLIT + reduce_add -- the first true reduction in this project.  ``mlp down``
is 4096->1024: the dimension that is fat is K, not N. Splitting K makes every
engine compute a PARTIAL SUM over its half, and the partials must be ADDED.
That is a real cross-engine reduction and it is deliberately NOT hidden --
:meth:`reduce_add` is a named call the caller writes out.

  Cost, stated plainly: barrier + ONE ``M*N`` bf16 ``eltwise_add`` per extra
  engine, executed on the primary while the workers idle, + barrier. At
  M=64, N=1024 that is 64Ki elements of add against 64*4096*1024 = 256 MFLOP
  of matmul, i.e. ~0.03% of the arithmetic -- but it is serialized, so the real
  cost is the two rendezvous plus the primary's non-overlapped add. It is only
  worth it because the matmul either side of it is 4 orders of magnitude bigger.

  K-SPLIT NEEDS HOST-SIDE PRE-SLICED WEIGHTS. This is the asymmetry with
  N-splitting and it is not fixable by address arithmetic: B is ``N x K``
  row-major, so a K-slice is a COLUMN slice of B's rows -- strided, one gap per
  row. :meth:`split_k` gives the element ranges; the caller uploads
  ``B[:, k0:k0+Kc].contiguous()`` (and, quantized, re-quantizes that slice) as
  its own blob per engine. For a model that is one-time weight prep at load, so
  it costs nothing at runtime, but it does cost DRAM: the down-projection
  weight is stored sliced instead of whole.

BARRIER = SYMMETRIC RENDEZVOUS
------------------------------
The ISA's only cross-engine primitive is a 1-bit per-engine flag with
SET / CLEAR / CHECK, where CHECK spin-waits for ``== 1`` (no wait-for-zero, no
DRAM-load + conditional branch, so a monotone phase-counter barrier is not
expressible). The shape that provably RE-ARMS inside one instruction stream is
the symmetric rendezvous validated by ``flag_rendezvous_repeat_test`` in
user_hw_test.py (27 rounds, exact payloads)::

    per engine:  FLAG_SET ; FLAG_CHECK(every other engine) ; <margin> ; FLAG_CLEAR

Every engine both signals and waits, so an asymmetric "primary sets go /
workers wait" scheme -- which cannot re-arm without a stale-1 window -- is not
used here. CLEAR placement is the delicate part and cuts both ways:

  * CLEAR too early -> the partner misses our SET -> hang.
  * CLEAR too late  -> the partner's NEXT rendezvous sees our stale SET ->
                       it races ahead and reads data we have not written.

The proven test buys margin by putting CLEAR one DMA after the CHECK. This
module emits ``barrier_margin_nops`` NOPs (default 32) between CHECK and CLEAR
for the same purpose; the window on the other side is a whole region body or a
whole single-engine region, i.e. orders of magnitude larger. Multi-region
programs are therefore supported directly -- see
``sharded_scheduler_multi_region_test``.

FULL-DUPLEX (FOUR-PHASE) RENDEZVOUS -- RECOMMENDED FOR NEW CODE
---------------------------------------------------------------
The paragraph above describes the ISA AS IT WAS. It has since grown
``FLAG_MODE_CHECK_CLEAR`` (spin-wait for ``== 0``), so "wait-for-zero is not
expressible" is no longer true, and the NOP margin it forced is no longer the
only option. :meth:`MultiEngineScheduler.release` / :meth:`~.join` implement
the shape that replaces it -- an ASYMMETRIC master/worker handshake in which
every flag EDGE is acknowledged before the next edge in that direction::

    master:  SET | work | CHECK_SET(all W) | CLEAR | CHECK_CLEAR(all W)
    worker:  CHECK_SET(0) | work | SET     | CHECK_CLEAR(0) | CLEAR

    1. M:0->1  "go"            workers are waiting in CHECK_SET(0)
    2. W:0->1  "I am done"     master is waiting in CHECK_SET(W)
    3. M:1->0  "round closed"  workers are waiting in CHECK_CLEAR(0)
    4. W:1->0  "I am re-armed" master is waiting in CHECK_CLEAR(W)

The master cannot reach the next round's SET without having seen every worker
CLEAR (4), so a stale worker flag can never satisfy its CHECK_SET; a worker
cannot reach its next CHECK_SET without having seen the master CLEAR (3), so a
stale master flag can never release it twice. That is a CORRECTNESS ARGUMENT,
not a timing margin: it needs no delay, and it does not care how uneven the
shards are or how little work the master has between rounds.

WHY THIS MATTERS AS THE ENGINE COUNT GROWS. The margin-based barrier couples
two bounds that squeeze from opposite directions::

    worker completion skew  <  margin  <  the master's next-round work

The right-hand side SHRINKS as cores are added, because the master's own shard
gets smaller. Measured on gemma3-1B decode: the margin scheme passed at 2
engines and failed at 4 and 8 -- silently, as finite-but-wrong logits, because
a fast worker re-read its own round's still-asserted release and ran the next
round against activations the master had not produced yet. The four-phase
handshake decouples them entirely and was verified on hardware at 8 engines.

Both shapes are available and both are supported. Choose by TOPOLOGY, not by
preference:

  * :meth:`~.release` / :meth:`~.join` -- master/worker. The master emits into
    the model's own stream and the workers run their own persistent program
    (see :meth:`~.emit_worker_program`). O(N) checks on the master, exactly one
    on each worker. This is the recommended shape for anything new.
  * :meth:`~.barrier` -- symmetric, all engines emitted in lockstep by one
    region body. Still the right primitive for :meth:`~.sharded_region` and
    friends, and still the default for existing callers. Pass
    ``handshake="four_phase"`` to the constructor to get the margin-free
    mechanism under the SAME symmetric contract; ``"nops"`` (the default)
    keeps the exact instruction stream the current models are validated
    against.

THE TWO SHAPES DO NOT INTEROPERATE INSIDE ONE PROGRAM. A master sitting in
``join()`` while a worker sits in a symmetric ``barrier()`` deadlocks, and
FLAG_CHECK has no timeout. The scheduler therefore LATCHES its rendezvous mode
on first use and refuses a mix (see ``_latch_rendezvous``).
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Optional

import user_dma_core
from user_dma_core import UnifiedEngine


class TensorRole(Enum):
    """How a DRAM buffer behaves under row sharding. See module docstring."""
    SHARED_ROWS = "shared_rows"
    SHARED_FULL = "shared_full"
    PER_ENGINE = "per_engine"


# Ops proven safe to run row-sharded with zero mid-chain synchronization
# (row-independent over M). Anything else must leave the sharded region.
SHARDED_OP_ALLOWLIST = frozenset({
    # matmul family
    "matmat_mul_core",
    "matmat_mul_core_dynamic",
    "matmat_mul_core_legacy",
    "quantized_matmat_mul_core",
    # Standalone pointwise activation implemented through matmat_mul_core with
    # a shared read-only identity matrix. Each M row remains independent.
    "activation_core",
    # norms
    "layer_norm_core_dram",
    "layer_norm_core_dram_dynamic",
    "layer_norm_core_dram_legacy",
    "rms_norm_core_dram",
    "rms_norm_core_dram_dynamic",
    "rms_norm_core_dram_legacy",
    # eltwise / activations
    "eltwise_core_dram",
    "eltwise_core_dram_dynamic",
    "eltwise_core_dram_legacy",
})

# Scalar / register plumbing a body legitimately needs; emits no data movement.
_SCALAR_HELPER_ALLOWLIST = frozenset({
    "alloc_isa_reg",
    "release_isa_reg",
    "generate_instruction_add_set",
    "float_to_bf19",
})


# Hardware alignment contract (see reference_hardware_behaviors):
#   * SRAM rows are 128 bytes; every DMA base and stride must respect that.
#   * AXI beats are 32 bytes; that is the absolute floor.
#   * The matvec unit consumes N in UE_VECTOR_SIZE (64) element vectors, which
#     is also the granularity the quantization scale blob is blocked at.
SRAM_ROW_BYTES = 128
AXI_BEAT_BYTES = 32
COL_ALIGN = user_dma_core.UE_VECTOR_SIZE   # 64 elements == 128 bytes at bpe=2

# UnifiedEngine.init_unified_engine() dma_writes 8192 uint16 (16 KB) of random
# data to the hardcoded DRAM_START_ADDR on EVERY engine construction. We guard
# 64 KB -- 4x the observed footprint -- so a change to the self-test's size
# cannot silently re-open the hole.
DRAM_SELFTEST_GUARD_BYTES = 0x10000


def _shifted(base_addr: int, offset_bytes: int, what: str) -> int:
    """``base + offset`` with the 128-byte SRAM-row contract asserted.

    Sharding contributes the OFFSET, so the offset itself must be a whole
    number of 128-byte SRAM rows -- that is the invariant this module owns and
    it is asserted unconditionally. The absolute address then inherits whatever
    the caller's allocator gave the base: if the base is 128-byte aligned the
    result is too (asserted), and a base that is merely 64-byte aligned (the
    ``allocate_tensor_dram`` default) still has to clear the 32-byte AXI beat.

    Every per-column address in this module goes through here; none is
    hand-rolled at a call site.
    """
    assert offset_bytes >= 0, f"{what}: negative offset {offset_bytes}"
    assert offset_bytes % SRAM_ROW_BYTES == 0, (
        f"{what}: shard offset {offset_bytes} B is not a multiple of the "
        f"{SRAM_ROW_BYTES} B SRAM row. A column shard must be a multiple of "
        f"{COL_ALIGN} elements; at bpe=2 that is exactly {COL_ALIGN * 2} B."
    )
    addr = base_addr + offset_bytes
    assert addr % AXI_BEAT_BYTES == 0, (
        f"{what}: address 0x{addr:x} is not {AXI_BEAT_BYTES} B AXI-beat aligned "
        f"(base 0x{base_addr:x} is misaligned to begin with)"
    )
    if base_addr % SRAM_ROW_BYTES == 0:
        assert addr % SRAM_ROW_BYTES == 0, (
            f"{what}: address 0x{addr:x} broke {SRAM_ROW_BYTES} B SRAM-row "
            f"alignment that base 0x{base_addr:x} had"
        )
    return addr


def capture_digest(ue) -> str:
    """SHA-256 over an engine's captured instruction words.

    Used by the num_engines==1 byte-identity assertion: the passthrough path
    must produce a stream indistinguishable from hand-written single-engine
    emission.
    """
    h = hashlib.sha256()
    for inst in ue.capture_buffer:
        for w in inst.words:
            h.update(int(w).to_bytes(16, "little", signed=int(w) < 0))
    return h.hexdigest()


# ==========================================================================
# NEW (master/worker sharding): the PRIVATE LOW-DRAM map
# ==========================================================================
# WHY A SECOND MAP EXISTS. The original worker arena is
# ``DRAM_START_ADDR + worker_dram_stride`` -- i.e. INSIDE the 0x8000_0000
# window a model lays its own params/tensor/program cursors into. Every worker
# allocation there is one arithmetic slip away from landing on live weights,
# and two shipped models already hand-roll their own base to dodge it. This map
# takes the opposite approach: it claims the LOW 2 GB, BELOW ``DRAM_START_ADDR``,
# which no single-engine model build ever touches, so a worker arena CANNOT
# alias the main map however the model moves its cursors.
#
#     stride = align_down(DRAM_START_ADDR / num_engines, 16 MB)
#     engine i -> [i * stride, (i + 1) * stride)
#         weights : base                         .. base + stride - 0x0200_0000
#         ISA     : base + stride - 0x0200_0000  .. base + stride - 0x0100_0000
#         tensor  : base + stride - 0x0100_0000  .. base + stride
#
# The window SCALES with the core count, which is the direction that matters:
# fewer cores means fewer, larger shards and a correspondingly larger arena
# each. The stride is rounded DOWN to 16 MB, so a core count that does not
# divide evenly leaves slack at the top rather than ever crossing 0x8000_0000.
#
# Opt in with ``MultiEngineScheduler(..., worker_map="private_low")``. The
# default stays ``"legacy"`` so existing callers keep the exact map they are
# validated against.
PRIVATE_ALIGN = 0x0100_0000           # 16 MB window granularity
PRIVATE_TENSOR_BYTES = 0x0100_0000    # 16 MB default, the ending slice of each window
PRIVATE_ISA_BYTES = 0x0100_0000       # 16 MB default, immediately before the tensor slice
MAX_PRIVATE_ENGINES = 16
"""Ceiling for the private map -- the per-engine FLAG index is 4 bits, so 16 is
the hard architectural limit; the practical one is whatever stride the arena
leaves per engine."""

# WHERE THE ARENA LIVES IS A PARAMETER, NOT A CONSTANT. The low 2 GB is only the
# DEFAULT, and it is the right default for a model whose own map starts at
# DRAM_START_ADDR (gemma3). A model that has re-based ITS OWN map into the low
# 2 GB (gemma4-e2b multi-core does exactly this) must carve its per-engine
# windows somewhere else, or the windows land on its weights -- which is the
# very failure this map exists to prevent, merely inverted. Pass an explicit
# ``(base, size)`` in that case; :class:`PrivateArena` owns the arithmetic
# either way.


def private_total() -> int:
    """Default arena size: everything below the main map at DRAM_START_ADDR."""
    return user_dma_core.DRAM_START_ADDR


def private_stride(num_engines: int, arena_bytes: Optional[int] = None,
                   isa_bytes: int = PRIVATE_ISA_BYTES,
                   tensor_bytes: int = PRIVATE_TENSOR_BYTES) -> int:
    """Bytes per engine when ``arena_bytes`` is split ``num_engines`` ways.

    Rounded DOWN to :data:`PRIVATE_ALIGN`, so an engine count that does not
    divide the arena evenly leaves slack at the TOP rather than overrunning it.
    """
    if arena_bytes is None:
        arena_bytes = private_total()
    if not 1 <= num_engines <= MAX_PRIVATE_ENGINES:
        raise ValueError(
            f"num_engines must be in [1, {MAX_PRIVATE_ENGINES}] for the private map, "
            f"got {num_engines}")
    stride = (arena_bytes // num_engines) & ~(PRIVATE_ALIGN - 1)
    floor = tensor_bytes + isa_bytes + PRIVATE_ALIGN
    if stride < floor:
        raise ValueError(
            f"num_engines={num_engines} over a 0x{arena_bytes:X} arena leaves only "
            f"0x{stride:X} per window, below the 0x{floor:X} needed for ISA + tensor "
            f"+ at least one weight block")
    assert num_engines * stride <= arena_bytes, "private windows must not leave the arena"
    return stride


def private_weight_bytes(num_engines: int, arena_bytes: Optional[int] = None,
                         isa_bytes: int = PRIVATE_ISA_BYTES,
                         tensor_bytes: int = PRIVATE_TENSOR_BYTES) -> int:
    """Weight arena inside one private window (the window minus its ISA + tensor tail)."""
    return (private_stride(num_engines, arena_bytes, isa_bytes, tensor_bytes)
            - tensor_bytes - isa_bytes)


@dataclass(frozen=True)
class PrivateRegion:
    """One engine's private window: weights, then its ISA slice, then tensor scratch."""

    engine_idx: int
    base: int
    weight_base: int
    weight_limit: int
    isa_base: int
    tensor_base: int

    @property
    def weight_capacity(self) -> int:
        return self.weight_limit - self.weight_base

    def describe(self) -> str:
        return (f"core {self.engine_idx}: 0x{self.base:08X} weights "
                f"{self.weight_capacity / 2**20:6.0f} MB  isa 0x{self.isa_base:08X}  "
                f"tensor 0x{self.tensor_base:08X}")


def private_region(engine_idx: int, num_engines: int, arena_base: int = 0,
                   arena_bytes: Optional[int] = None,
                   isa_bytes: int = PRIVATE_ISA_BYTES,
                   tensor_bytes: int = PRIVATE_TENSOR_BYTES) -> PrivateRegion:
    """The private window belonging to ``engine_idx`` when ``num_engines`` are in play.

    Layout inside a window, low to high: WEIGHTS, then the ISA slice, then the
    TENSOR slice at the very top. The two fixed-size slices sit at the END so
    the weight arena -- the only part whose size varies with the engine count --
    grows and shrinks against the window base, leaving the ISA and scratch
    addresses at a constant offset from the window top.
    """
    if not 0 <= engine_idx < num_engines:
        raise ValueError(f"engine_idx {engine_idx} outside [0, {num_engines})")
    stride = private_stride(num_engines, arena_bytes, isa_bytes, tensor_bytes)
    base = arena_base + engine_idx * stride
    tensor_base = base + stride - tensor_bytes
    isa_base = tensor_base - isa_bytes
    return PrivateRegion(engine_idx=engine_idx, base=base, weight_base=base,
                         weight_limit=isa_base, isa_base=isa_base, tensor_base=tensor_base)


def describe_private_map(num_engines: int, arena_base: int = 0,
                         arena_bytes: Optional[int] = None,
                         isa_bytes: int = PRIVATE_ISA_BYTES,
                         tensor_bytes: int = PRIVATE_TENSOR_BYTES) -> str:
    if arena_bytes is None:
        arena_bytes = private_total()
    stride = private_stride(num_engines, arena_bytes, isa_bytes, tensor_bytes)
    lines = [f"  private DRAM map over "
             f"[0x{arena_base:08X}..0x{arena_base + arena_bytes:08X}), "
             f"{num_engines} core(s), {stride / 2**20:.0f} MB/core:"]
    lines += ["    " + private_region(i, num_engines, arena_base, arena_bytes,
                                      isa_bytes, tensor_bytes).describe()
              for i in range(num_engines)]
    return "\n".join(lines)


class PrivateArena:
    """Per-engine private DRAM windows, and the bump allocators inside them.

    THE ALLOCATOR IS SHARED, DELIBERATELY. A model needs these addresses in
    places that have no scheduler in hand -- laying out per-engine attention
    scratch during tensor init, bounds-checking a worker's ISA image at program
    write time -- while the scheduler needs the same windows to place worker
    engines and weight shards. Two independent allocators over one address range
    is precisely how a worker's program ends up on top of another engine's
    scratch, so there is ONE arena object and everything asks it.

    Build it once on the model, hand it to every
    :class:`MultiEngineScheduler` for the run (``arena=``), and the schedulers
    stop owning any DRAM arithmetic of their own.
    """

    def __init__(self, num_engines: int, arena_base: int = 0,
                 arena_bytes: Optional[int] = None,
                 isa_bytes: int = PRIVATE_ISA_BYTES,
                 tensor_bytes: int = PRIVATE_TENSOR_BYTES,
                 external_isa: Optional[tuple] = None,
                 verbose: bool = False):
        """``external_isa=(base, stride)`` moves the per-engine ISA slices OUT of
        the private windows and into a region the model owns elsewhere.

        Worth doing when private space is the scarce resource and the model map
        is not: worker programs are a few MB, so a 16 MB slice per window buys
        little and costs the weight arena the same 16 MB on every core. With it
        set, a window is just [ weights | tensor ] and the whole reclaimed slice
        goes to weights.
        """
        if arena_bytes is None:
            arena_bytes = private_total()
        assert arena_base % PRIVATE_ALIGN == 0, (
            f"arena_base 0x{arena_base:X} must be {PRIVATE_ALIGN // 2**20} MB aligned")
        self.num_engines = num_engines
        self.arena_base = arena_base
        self.arena_bytes = arena_bytes
        self.external_isa = external_isa
        # What the WINDOW reserves for ISA (0 when the slices live elsewhere)
        # versus how big one engine's ISA slice IS -- the same number only when
        # the slices are carved from the windows.
        self._carve_isa_bytes = 0 if external_isa is not None else isa_bytes
        self.isa_bytes = external_isa[1] if external_isa is not None else isa_bytes
        self.tensor_bytes = tensor_bytes
        self.stride = private_stride(num_engines, arena_bytes,
                                     self._carve_isa_bytes, tensor_bytes)
        self.regions = [private_region(i, num_engines, arena_base, arena_bytes,
                                       self._carve_isa_bytes, tensor_bytes)
                        for i in range(num_engines)]
        if external_isa is not None:
            ext_base, ext_stride = external_isa
            assert ext_base % 64 == 0, "external ISA base must be 64 B aligned"
            self.regions = [replace(r, isa_base=ext_base + i * ext_stride)
                            for i, r in enumerate(self.regions)]
        self._weight_cursor = [r.weight_base for r in self.regions]
        self._tensor_cursor = [r.tensor_base for r in self.regions]
        if verbose:
            print(self.describe())

    # -- introspection ------------------------------------------------------
    def region(self, engine_idx: int) -> PrivateRegion:
        return self.regions[engine_idx]

    def isa_base(self, engine_idx: int) -> int:
        return self.regions[engine_idx].isa_base

    def isa_limit(self, engine_idx: int) -> int:
        return self.regions[engine_idx].isa_base + self.isa_bytes

    def weight_bytes(self) -> int:
        return self.stride - self._carve_isa_bytes - self.tensor_bytes

    def usage(self) -> list[int]:
        """Bytes of weight arena used per engine."""
        return [self._weight_cursor[i] - self.regions[i].weight_base
                for i in range(self.num_engines)]

    def describe(self) -> str:
        if self.external_isa is None:
            return describe_private_map(self.num_engines, self.arena_base,
                                        self.arena_bytes, self.isa_bytes,
                                        self.tensor_bytes)
        # Built from self.regions, not recomputed: the ISA bases were relocated
        # after the carve and describe_private_map would recompute the old ones.
        b, st = self.external_isa
        lines = [f"  Private map: 0x{self.arena_base:08X} .. "
                 f"+{self.arena_bytes / 2**20:.0f} MB, "
                 f"{self.num_engines} core(s), {self.stride / 2**20:.0f} MB/core:"]
        lines += ["    " + r.describe() for r in self.regions]
        lines.append(f"    ISA slices are OUTSIDE the private map: "
                     f"0x{b:08X} + {st / 2**20:.0f} MB/core "
                     f"({st * self.num_engines / 2**20:.0f} MB total)")
        return "\n".join(lines)

    # -- allocation ---------------------------------------------------------
    def alloc_weights(self, engine_idx: int, size_bytes: int, what: str) -> int:
        """Bump-allocate in an engine's WEIGHT arena, 64 B aligned."""
        addr = (self._weight_cursor[engine_idx] + 63) & ~63
        end = addr + size_bytes
        limit = self.regions[engine_idx].weight_limit
        if end > limit:
            raise MemoryError(
                f"{what}: engine {engine_idx} private weight arena overflow -- needs "
                f"0x{end:X}, window ends at 0x{limit:X} "
                f"({self.weight_bytes() // 2**20} MB per core at "
                f"num_engines={self.num_engines})")
        self._weight_cursor[engine_idx] = end
        return addr

    def alloc_tensor(self, engine_idx: int, size_bytes: int, what: str) -> int:
        """Bump-allocate in an engine's TENSOR window, 64 B aligned.

        This is where anything an engine must not share goes: attention scratch,
        per-engine staging buffers -- the things that are WRITTEN, as opposed to
        the read-only inputs every engine can address in the primary's map.
        """
        addr = (self._tensor_cursor[engine_idx] + 63) & ~63
        limit = self.regions[engine_idx].tensor_base + self.tensor_bytes
        if addr + size_bytes > limit:
            raise MemoryError(
                f"{what}: engine {engine_idx} private tensor window overflow -- needs "
                f"0x{addr + size_bytes:X}, window ends at 0x{limit:X} "
                f"({self.tensor_bytes // 2**20} MB per core)")
        self._tensor_cursor[engine_idx] = addr + size_bytes
        return addr

    def alloc_tensor_all(self, size_bytes: int, what: str) -> list[int]:
        """One same-sized private buffer per engine; returns them indexed by engine.

        The common shape for per-engine scratch: every engine needs its own copy
        of the same thing, and the caller wants a list it can index by engine.
        """
        return [self.alloc_tensor(i, size_bytes, what) for i in range(self.num_engines)]

    # -- protection ---------------------------------------------------------
    def check_isa_fits(self, engine_idx: int, addr: int, size_bytes: int) -> None:
        """Refuse a program image that would leave its engine's ISA slice.

        The ISA slice sits directly below the tensor slice, which sits directly
        below the NEXT engine's window. An overrun would not fault -- it would
        silently scribble instructions over a neighbour's scratch or weights, so
        it has to be caught where the image is written.
        """
        region = self.regions[engine_idx]
        limit = self.isa_limit(engine_idx)
        if addr < region.isa_base or addr + size_bytes > limit:
            if self.external_isa is not None:
                spill = "next engine's ISA slice or the model map"
            else:
                spill = ("tensor slice"
                         if addr + size_bytes <= region.tensor_base + self.tensor_bytes
                         else "next engine window")
            raise MemoryError(
                f"engine {engine_idx} ISA overflow: program "
                f"[0x{addr:X}..0x{addr + size_bytes:X}) is outside its slice "
                f"[0x{region.isa_base:X}..0x{limit:X}) "
                f"({self.isa_bytes // 2**20} MB). Writing it would corrupt the {spill}.")

    def verify(self, engines: Optional[list] = None, verbose: bool = True) -> None:
        """Check every engine's weight arena (and, if given, ISA cursor) is in bounds.

        Cheap, host-side, and worth calling after setup: an overflow here is
        silent corruption of a neighbouring engine's memory, not a fault.
        """
        for i, region in enumerate(self.regions):
            used = self._weight_cursor[i] - region.weight_base
            cap = region.weight_capacity
            if self._weight_cursor[i] > region.weight_limit:
                raise MemoryError(
                    f"engine {i} weight arena overflow: {used / 2**20:.1f} MB used of "
                    f"{cap / 2**20:.1f} MB, spilling into its ISA slice at "
                    f"0x{region.isa_base:X}")
            isa_used = None
            if engines is not None and i < len(engines) and engines[i] is not None:
                isa_used = engines[i].get_program_dram_addr() - region.isa_base
                if i > 0 and not 0 <= isa_used <= self.isa_bytes:
                    raise MemoryError(
                        f"engine {i} ISA cursor "
                        f"0x{engines[i].get_program_dram_addr():X} is outside its "
                        f"{self.isa_bytes // 2**20} MB slice at 0x{region.isa_base:X}")
            if verbose:
                tensor_used = self._tensor_cursor[i] - region.tensor_base
                isa_txt = (f"{isa_used / 1024:7.1f} KB" if isa_used is not None and i > 0
                           else "     (main map)")
                print(f"    core {i}: weights {used / 2**20:6.1f} / {cap / 2**20:.0f} MB"
                      f"   isa {isa_txt} / {self.isa_bytes // 2**20} MB"
                      f"   tensor {tensor_used / 2**20:5.2f} / "
                      f"{self.tensor_bytes // 2**20} MB")


def can_split(N: int, num_engines: int) -> bool:
    """Whether ``N`` columns can be handed to ``num_engines`` at 64-column granularity.

    False when N has fewer than one 64-block per engine. A caller uses this to DECIDE
    whether to shard an op at all rather than to discover it as an assertion: gemma3's
    K/V projections are N=256, i.e. 4 blocks, so they cap out at 4 engines however many
    the board has, while the rest of the decoder shards to 8.
    """
    return num_engines == 1 or (N % COL_ALIGN == 0 and N // COL_ALIGN >= num_engines)


def max_shards(N: int) -> int:
    """How many engines ``N`` columns can actually feed at 64-column granularity.

    The companion to :func:`can_split` for callers that would rather shard over a
    subset than not shard at all: pass this as ``max_engines``.
    """
    return N // COL_ALIGN if N % COL_ALIGN == 0 else 0


class _DenseBF16:
    """Sentinel ``data_type`` for an UNQUANTIZED bf16 weight blob.

    Not a member of ``TYPE``: that enum is the hardware's quantization-format
    field, and bf16 weights are the absence of one -- they carry no scale blob
    and go to ``matmat_mul_core`` rather than ``quantized_matmat_core``. It
    exists so the column materializer can size and slice a dense blob with the
    same code path as a quantized one.
    """

    def __repr__(self) -> str:
        return "DENSE_BF16"


DENSE_BF16 = _DenseBF16()


def _weight_elem_bytes(data_type) -> float:
    """Bytes per weight element for a weight blob (IF4 packs two per byte)."""
    if data_type == user_dma_core.TYPE.IF4:
        return 0.5
    if data_type == user_dma_core.TYPE.IF8:
        return 1.0
    if data_type is DENSE_BF16:
        return 2.0
    raise AssertionError(
        f"materialized weight shards support IF4/IF8/DENSE_BF16 only, got "
        f"{data_type!r}.")


# ==========================================================================
# NEW (master/worker sharding): MATERIALIZED weight shards
# ==========================================================================
# TWO WAYS TO GIVE AN ENGINE ITS COLUMN BLOCK, and they are complementary:
#
#   ZERO-COPY (ColumnShardContext.b_addr / .scale_addr) -- pass a SHIFTED
#   address into the one shared weight blob. Nothing is duplicated. Right for a
#   region emitted once, where the address is a compile-time constant.
#
#   MATERIALIZED (:meth:`MultiEngineScheduler.shard_quantized_weight`, below) --
#   COPY each engine's column block into that engine's private arena, packed.
#   Costs DRAM and a one-time load-side copy, and buys the thing the zero-copy
#   path cannot express: a PACKED PER-LAYER STRIDE. In the shared blob, layer L
#   of engine i's shard is at ``base + L*LAYER_SIZE + col_offset*K*bpe`` -- two
#   terms, so a hardware-looped body would need to recompute the column term
#   every iteration. In a private arena only this engine's columns exist, laid
#   out layer after layer, so the stride is a single ``add_imm`` on a runtime
#   cursor and one captured body can serve all L layers.
@dataclass
class WeightShard:
    """One engine's column block of one sharded weight, in that engine's private DRAM.

    ``weight_addr`` / ``scale_addr`` are LAYER 0; layer L is at
    ``weight_addr + L * layer_stride``. The stride is THIS shard's own packed size,
    not the source image's per-layer size.
    """

    engine_idx: int
    col_offset: int
    cols: int
    weight_addr: int
    scale_addr: int
    layer_stride: int
    scale_layer_stride: int


@dataclass
class ShardedWeight:
    """A full [N, K] quantized weight distributed column-wise over the engines."""

    name: str
    K: int
    N: int
    layers: int
    data_type: object
    shards: list = field(default_factory=list)

    def shard(self, engine_idx: int) -> WeightShard:
        return self.shards[engine_idx]

    def shard_or_none(self, engine_idx: int) -> Optional[WeightShard]:
        """This engine's block, or None when the weight was too narrow to reach it.

        A weight sharded with ``max_engines`` covers only engines 0..max-1; the
        engines past the end still run the round, they just emit nothing for it.
        """
        return self.shards[engine_idx] if engine_idx < len(self.shards) else None

    def summary(self) -> str:
        parts = ", ".join(f"e{s.engine_idx}:{s.cols}" for s in self.shards)
        return (f"{self.name}: N={self.N} over {len(self.shards)} engine(s) [{parts}] "
                f"K={self.K}, {self.layers} layer(s)")


# ==========================================================================
# NEW (master/worker sharding): BATCH-split decode attention
# ==========================================================================
# Distinct from :class:`HeadShardContext`, which splits PREFILL flash attention
# by head. This splits ``unified_attention_core`` by its BATCH (query-row)
# dimension, which is what a decode step has: one token, a handful of query
# rows against a shared KV cache. Two differences drive the separate type:
#   * the sequence length is a RUNTIME value (it grows every token), so the
#     per-engine bias row offset must be computed ON DEVICE, not baked in; and
#   * the core stages V-transpose / scores / scaled-Q through a SCRATCH buffer,
#     so every engine needs its own out of private tensor space.
@dataclass
class AttentionShard:
    """One engine's slice of a batch-split unified_attention_core."""

    engine_idx: int
    batch_offset: int
    batch_rows: int
    scratch_addr: int


@dataclass
class AttentionOp:
    """An attention op inside a worker round: everything needed to re-emit the core.

    Addresses are the PRIMARY's tensor buffers -- Q, bias and OUT are sliced by this
    engine's batch offset, K/V are shared whole (GQA with a shared KV head), and only
    SCRATCH is private.
    """

    sa: "ShardedAttention"
    q_addr: int
    k_addr: int
    v_addr: int
    bias_addr: int
    out_addr: int
    identity_addr: int
    kv_layer_stride: int
    scale_bf16: int


@dataclass
class ShardedAttention:
    """A unified_attention_core distributed over its batch (query-row) dimension."""

    name: str
    batch: int
    head_dim: int
    aligned_seq_len: int
    shards: list = field(default_factory=list)

    def shard(self, engine_idx: int) -> AttentionShard:
        return self.shards[engine_idx]

    def summary(self) -> str:
        parts = ", ".join(f"e{s.engine_idx}:{s.batch_rows}" for s in self.shards)
        return (f"{self.name}: batch={self.batch} over {len(self.shards)} engine(s) "
                f"[{parts}] head_dim={self.head_dim}")


def _const_addr(ue, literal: int, scratch_reg: int) -> int:
    """Load a literal DRAM address into a GPR; returns the register index."""
    ue.generate_instruction_add_set(scratch_reg, user_dma_core.ue_35bit_addr_shifter(literal))
    return scratch_reg


def _offset_addr(ue, gpr_off: int, literal: int, scratch_reg: int) -> int:
    """``scratch = gpr_off + literal`` in 35-bit word-address space."""
    ue.generate_instruction_add_imm(
        src_reg_idx=gpr_off,
        immediate_value=user_dma_core.ue_35bit_addr_shifter(literal),
        dst_reg_idx=scratch_reg)
    return scratch_reg


class _ShardedEngineProxy:
    """Restricts a sharded body to the proven-safe op set (requirement 7).

    A mistake becomes a compile-time ``AssertionError`` instead of a silent
    wrong answer. ``ctx.unsafe_ue`` is the deliberate escape hatch.
    """

    def __init__(self, ue):
        object.__setattr__(self, "_ue", ue)

    def __getattr__(self, name):
        ue = object.__getattribute__(self, "_ue")
        if name in SHARDED_OP_ALLOWLIST or name in _SCALAR_HELPER_ALLOWLIST:
            return getattr(ue, name)
        raise AssertionError(
            f"{name!r} is not allowed inside a sharded region. Only row-independent "
            f"ops are sharded in v1: {sorted(SHARDED_OP_ALLOWLIST)}. "
            f"Attention, strided SRAM<->DRAM marshalling, permutes and RoPE must run "
            f"single-engine: close the region (end_sharded()/leave sharded_region()), "
            f"emit them on the primary engine, then open a new region. "
            f"If you really mean it, use ctx.unsafe_ue."
        )

    def __setattr__(self, name, value):  # pragma: no cover - defensive
        raise AssertionError("sharded-region engine proxy is read-only")


class ShardContext:
    """Per-engine view handed to a sharded-region body.

    Attributes:
        ue:         op-allowlisted engine proxy (emit ops on this)
        unsafe_ue:  the raw UnifiedEngine (escape hatch)
        engine_idx: 0 == primary
        M:          full row count of the region
        rows:       THIS engine's row count
        row_offset: THIS engine's first row within M
        is_primary: engine_idx == 0
    """

    def __init__(self, scheduler, engine_idx: int, ue, M: int, row_offset: int, rows: int):
        self._sched = scheduler
        self.engine_idx = engine_idx
        self.unsafe_ue = ue
        self.ue = _ShardedEngineProxy(ue)
        self.M = M
        self.rows = rows
        self.row_offset = row_offset
        self.is_primary = engine_idx == 0
        self._m_reg = None

    # -- addressing ---------------------------------------------------------
    def rows_addr(self, base_addr: int, per_row_bytes: int) -> int:
        """Address of THIS engine's row block inside a SHARED_ROWS buffer.

        ``per_row_bytes`` is the full row pitch of the buffer (e.g. N*2 for a
        [M, N] bf16 tensor). This is the one piece of arithmetic call sites
        must never hand-roll.
        """
        assert per_row_bytes > 0, "per_row_bytes must be positive"
        return base_addr + self.row_offset * per_row_bytes

    # Explicit alias so a call site can state the role it means.
    def addr(self, base_addr: int, role: TensorRole = TensorRole.SHARED_FULL,
             per_row_bytes: Optional[int] = None) -> int:
        if role is TensorRole.SHARED_FULL:
            assert per_row_bytes is None, "SHARED_FULL takes no per_row_bytes"
            return base_addr
        if role is TensorRole.SHARED_ROWS:
            assert per_row_bytes is not None, "SHARED_ROWS requires per_row_bytes"
            return self.rows_addr(base_addr, per_row_bytes)
        raise AssertionError("PER_ENGINE buffers are resolved with ctx.per_engine(name)")

    def per_engine(self, name: str) -> int:
        """Resolve a PER_ENGINE (duplicated scratch) buffer for this engine."""
        return self._sched.per_engine_addr(name, self.engine_idx)

    # -- runtime row count --------------------------------------------------
    @property
    def m_reg(self) -> int:
        """GPR primed with THIS engine's row count (requirement 3).

        Lazily allocated + ``add_set`` at first use, so with num_engines == 1
        the emitted stream is exactly what hand-written code emits.
        """
        if self._m_reg is None:
            self._m_reg = self._sched._acquire_m_reg(self.engine_idx, self.rows)
        return self._m_reg

    def elems(self, per_row_elems: int) -> int:
        """Flat element count for this engine's rows (for eltwise ``size=``)."""
        return self.rows * per_row_elems


class ColumnShardContext:
    """Per-engine view handed to an N-sharded (output-column) region body.

    The M-mode analogue is :class:`ShardContext`; this one splits the OUTPUT
    COLUMNS instead of the rows. ``A`` is read in full by every engine (it is
    the same M x K activation), ``B``/scale/bias are sliced by row block, and
    the output goes to a per-engine contiguous ``[M, cols]`` buffer.

    Attributes:
        ue:         op-allowlisted engine proxy (emit ops on this)
        unsafe_ue:  the raw UnifiedEngine (escape hatch)
        engine_idx: 0 == primary
        N:          full output-column count of the region
        cols:       THIS engine's column count (always COL_ALIGN-aligned)
        col_offset: THIS engine's first column within N
        is_primary: engine_idx == 0
    """

    def __init__(self, scheduler, engine_idx: int, ue, N: int, col_offset: int, cols: int):
        self._sched = scheduler
        self.engine_idx = engine_idx
        self.unsafe_ue = ue
        self.ue = _ShardedEngineProxy(ue)
        self.N = N
        self.cols = cols
        self.col_offset = col_offset
        self.is_primary = engine_idx == 0
        self._n_reg = None

    # -- B / scale / bias slicing -------------------------------------------
    def b_addr(self, base_addr: int, K: int, data_type=None) -> int:
        """Address of THIS engine's row block of a ``B`` (N x K) weight.

        ``data_type=None`` means plain bf16 (2 B/element); pass ``TYPE.IF4`` or
        ``TYPE.IF8`` for a quantized blob, whose element widths are 0.5 B and
        1 B respectively -- the same widths ``matmat_mul_core_legacy`` uses for
        its own N-chunk offset (``user_dma_core.py:5517``).
        """
        assert K % COL_ALIGN == 0, (
            f"b_addr: K={K} must be a multiple of UE_VECTOR_SIZE={COL_ALIGN}")
        TYPE = user_dma_core.TYPE
        if data_type is None:
            offset = self.col_offset * K * 2
        elif data_type == TYPE.IF8:
            offset = self.col_offset * K
        elif data_type == TYPE.IF4:
            offset = (self.col_offset * K) // 2
        else:
            raise AssertionError(f"b_addr: unsupported data_type {data_type!r}")
        return _shifted(base_addr, offset, f"B shard (engine {self.engine_idx})")

    def scale_addr(self, base_addr: int, K: int) -> int:
        """Address of THIS engine's slice of a quantized-B scale blob.

        The blob is one bf16 per ``UE_VECTOR_SIZE`` block of the FLATTENED
        (N, K) weight, i.e. effectively ``[N, K/64]`` bf16, so a column shard
        is a contiguous row block of it -- the exact same slice shape as B.
        """
        assert K % COL_ALIGN == 0, (
            f"scale_addr: K={K} must be a multiple of UE_VECTOR_SIZE={COL_ALIGN}; "
            f"the scale DRAM stride is linear in whole K-blocks")
        offset = ((self.col_offset * K) // COL_ALIGN) * 2
        return _shifted(base_addr, offset, f"scale shard (engine {self.engine_idx})")

    def bias_addr(self, base_addr: int) -> int:
        """Address of THIS engine's slice of a ``bias_mode='broadcast_N'`` vector.

        A length-N bias LOOKS like a broadcast constant and is therefore the
        single most likely thing to be passed through unsliced. It is not
        broadcast over N -- it is indexed by N. Slice it.
        """
        return _shifted(base_addr, self.col_offset * 2,
                        f"broadcast_N bias shard (engine {self.engine_idx})")

    def col_out(self, name: str) -> int:
        """THIS engine's contiguous ``[M, cols]`` output buffer (see
        :meth:`MultiEngineScheduler.alloc_col_output`)."""
        return self._sched.col_output_addr(name, self.engine_idx)

    # -- runtime column count -----------------------------------------------
    @property
    def n_reg(self) -> int:
        """GPR primed with THIS engine's column count, for the PBI/dynamic path.

        Lazily allocated + ``add_set`` at first use so that with
        num_engines == 1 the emitted stream is exactly what hand-written code
        emits (mirrors ``ShardContext.m_reg``).
        """
        if self._n_reg is None:
            self._n_reg = self._sched._acquire_n_reg(self.engine_idx, self.cols)
        return self._n_reg

    def elems(self, rows: int) -> int:
        """Flat element count of this engine's ``[rows, cols]`` lane (eltwise ``size=``)."""
        return rows * self.cols


class KShardContext:
    """Per-engine view handed to a K-sharded (REDUCTION) region body.

    Unlike M- and N-sharding, this one does NOT partition the output: every
    engine emits the full ``[M, N]``, as a PARTIAL SUM over its slice of K.
    The partials are combined by :meth:`MultiEngineScheduler.reduce_add`.

    There is no ``b_addr``/``scale_addr`` here on purpose. ``B`` is ``N x K``
    row-major, so ``B[:, k0:k0+k_cols]`` is strided -- one gap per row -- and
    is NOT reachable by shifting a base address. The caller must upload a
    contiguous pre-sliced blob per engine (one-time weight prep in a model) and
    pass its address itself. Making that impossible to get wrong by accident is
    worth the small amount of ceremony.

    Attributes:
        K:          full reduction length of the region
        k_cols:     THIS engine's slice length (always COL_ALIGN-aligned)
        k_offset:   THIS engine's first K element
    """

    def __init__(self, scheduler, engine_idx: int, ue, K: int, k_offset: int, k_cols: int):
        self._sched = scheduler
        self.engine_idx = engine_idx
        self.unsafe_ue = ue
        self.ue = _ShardedEngineProxy(ue)
        self.K = K
        self.k_cols = k_cols
        self.k_offset = k_offset
        self.is_primary = engine_idx == 0
        self._k_reg = None

    @property
    def k_reg(self) -> int:
        """GPR primed with THIS engine's K slice length, for the PBI path."""
        if self._k_reg is None:
            self._k_reg = self._sched._acquire_n_reg(self.engine_idx, self.k_cols)
        return self._k_reg


class HeadShardContext:
    """Per-engine view handed to an ATTENTION-HEAD-sharded region body.

    Attention is the one stage that cannot be row- or column-sharded. Row
    sharding splits M (tokens), but a row of the softmax needs the WHOLE key
    axis, and -- decisively -- ``prefill_flash_attention_core`` builds ``V^T``
    into a single ``SCRATCH_VT`` per call, so two engines row-splitting one head
    would write the same scratch address. Heads, by contrast, are completely
    independent: disjoint Q slice in, disjoint output slice out, K/V read-only.

    THE ``ue`` HERE IS RAW, NOT THE ALLOWLIST PROXY. Every other context hands
    out ``_ShardedEngineProxy`` because its region body emits a handful of
    high-level ops (``matmat_mul_core``, ``eltwise_core_dram``, ...) that can be
    enumerated and policed. The flash kernels are NOT built from those: they
    emit ``start_queue_for_bf16_matvec_operation``,
    ``start_queue_for_bf16_softmax_operation``, ``broadcast_mul`` and bare
    SRAM/DRAM DMAs directly (``nn_lib.py:1012``). Allowlisting that set would
    permit essentially every method on the engine, which is not a guard rail at
    all -- so this region is deliberately less policed than the others, and the
    invariant it owns is address discipline instead: use the accessors below and
    never derive an attention address by hand.

    GQA: ``q_addr`` is indexed by Q head, ``kv_addr`` by KV head
    (``head_off // gqa_ratio``). Splitting on a KV-head boundary is enforced by
    :meth:`MultiEngineScheduler.split_heads`, so an engine never owns a partial
    KV group.

    Attributes:
        ue:         RAW UnifiedEngine (see above -- no allowlist here)
        engine_idx: 0 == primary
        H:          full Q-head count of the region
        heads:      THIS engine's Q-head count
        head_off:   THIS engine's first Q head within H
        gqa_ratio:  Q heads per KV head (1 == MHA)
        is_primary: engine_idx == 0
    """

    def __init__(self, scheduler, engine_idx: int, ue, H: int, head_off: int,
                 heads: int, gqa_ratio: int, seq_len: int, head_dim: int,
                 elem_bytes: int = 2):
        self._sched = scheduler
        self.engine_idx = engine_idx
        self.ue = ue                 # RAW: flash is raw primitives, not ops
        self.unsafe_ue = ue          # alias, so bodies read like the others
        self.H = H
        self.heads = heads
        self.head_off = head_off
        self.gqa_ratio = gqa_ratio
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.elem_bytes = elem_bytes
        self.is_primary = engine_idx == 0

    @property
    def head_bytes(self) -> int:
        """Bytes of one [seq_len, head_dim] head plane."""
        return self.seq_len * self.head_dim * self.elem_bytes

    @property
    def groups(self) -> int:
        """KV groups THIS engine touches (the last may be partial under
        ``mode="qheads"``). Prefer :meth:`call_runs`, which is exact."""
        first_kv = self.head_off // self.gqa_ratio
        last_kv = (self.head_off + self.heads - 1) // self.gqa_ratio
        return last_kv - first_kv + 1

    def call_runs(self) -> list[tuple[int, int, int]]:
        """Decompose this engine's head range into ``(q_head, kv_head, n_q)``
        kernel calls -- THE contract between the split and the kernel.

        Each run is a MAXIMAL span of consecutive Q heads sharing one KV head,
        which is exactly what one flash call can process (it re-reads
        ``K_DRAM_ADDR`` unshifted per head and builds ``V^T`` once). Whole-group
        ownership yields one run per group; a straddling range yields a short
        run at each ragged end and full runs in between. Both are correct, and
        the caller never has to know which case it is in.

        Head indices are ABSOLUTE (model-wide), so they feed the ``*_addr``
        accessors directly.
        """
        runs = []
        h = self.head_off
        end = self.head_off + self.heads
        while h < end:
            kv = h // self.gqa_ratio
            # Stop at whichever comes first: the end of this KV group, or the
            # end of what this engine owns.
            run_end = min((kv + 1) * self.gqa_ratio, end)
            runs.append((h, kv, run_end - h))
            h = run_end
        return runs

    # -- Q / K / V / output slicing -----------------------------------------
    # ``head``/``kv_head`` are ABSOLUTE indices, as produced by call_runs().
    def q_addr(self, base_addr: int, head: Optional[int] = None) -> int:
        """Q head ``head`` in a [H, seq_len, head_dim] blob (default: this
        engine's first). The kernel walks the run's remaining heads itself,
        which is correct only because Q heads of one KV group are contiguous."""
        head = self.head_off if head is None else head
        return _shifted(base_addr, head * self.head_bytes,
                        f"Q head {head} (engine {self.engine_idx})")

    def kv_addr(self, base_addr: int, kv_head: Optional[int] = None) -> int:
        """KV head ``kv_head`` in a [H/gqa_ratio, seq_len, head_dim] blob.

        Indexed by KV head, NOT Q head -- passing a Q index here is the single
        most likely GQA mistake and reads out of bounds on the last engine
        rather than failing loudly.
        """
        kv_head = (self.head_off // self.gqa_ratio) if kv_head is None else kv_head
        return _shifted(base_addr, kv_head * self.head_bytes,
                        f"KV head {kv_head} (engine {self.engine_idx})")

    def out_addr(self, base_addr: int, head: Optional[int] = None) -> int:
        """Output slice for Q head ``head``. Disjoint across engines AND runs,
        so the concatenated [H, seq_len, head_dim] is complete after one exit
        barrier."""
        head = self.head_off if head is None else head
        return _shifted(base_addr, head * self.head_bytes,
                        f"attn out head {head} (engine {self.engine_idx})")

    def bias_addr(self, base_addr: int, per_head: bool = False,
                  head: Optional[int] = None) -> int:
        """Attention bias / mask base.

        ``prefill_flash_attention_core`` indexes its bias ``((i+r) * N_q + j)``
        with NO head term (``nn_lib.py:1114``), i.e. it assumes ONE
        [seq_len, seq_len] plane shared by every head -- so the default here is
        to pass the base through unshifted. Pass ``per_head=True`` only for a
        model that really does upload [H, seq_len, seq_len]; note the kernel
        will still read only the plane at the address it is given, which is
        correct precisely because each engine gets its own head's plane.
        """
        if base_addr is None:
            return None
        if not per_head:
            return base_addr
        plane = self.seq_len * self.seq_len * self.elem_bytes
        head = self.head_off if head is None else head
        return _shifted(base_addr, head * plane,
                        f"attn bias head {head} (engine {self.engine_idx})")

    # -- scratch -------------------------------------------------------------
    def scratch(self, name: str = "attn_scratch") -> int:
        """THIS engine's PRIVATE flash scratch base.

        One pointer covers both buffers the kernel derives from it --
        ``SCRATCH_VT`` at +0 and ``SCRATCH_SM`` at ``+head_dim*seq_len*bpe``
        (``nn_lib.py:1024``) -- so registering one region of
        :meth:`MultiEngineScheduler.attn_scratch_bytes` is sufficient.
        Sharing this across engines is the failure mode this whole class exists
        to prevent: ``SCRATCH_SM`` is written per head and read back per head,
        so two engines on one base interleave and silently corrupt.
        """
        return self._sched.per_engine_addr(name, self.engine_idx)


class MultiEngineScheduler:
    """Emits row-sharded regions of an existing model's program onto N engines.

    Usage inside a model's ``compile_x()`` (the model is engine 0)::

        sched = MultiEngineScheduler(self, num_engines=2)
        self.start_capture()                 # model's own capture, untouched
        sched.begin_program()                # opens worker captures
        ...
        sched.sharded_region(S, body)        # replayed per engine
        ... single-engine attention on self ...
        sched.sharded_region(S, body2)
        ...
        sched.finalize()                     # closes/writes worker programs
        self.stop_capture(); ... model writes its own program ...

    and at execution time call :meth:`start_workers` before launching the
    primary program, every time (workers halt at the end of each run).
    """

    def __init__(self,
                 primary_ue: UnifiedEngine,
                 num_engines: int = 2,
                 *,
                 engine_base_stride: int = 0x00010000,
                 worker_dram_base: Optional[int] = None,
                 worker_dram_stride: int = 0x10000000,
                 worker_tensor_offset: int = 0x08000000,
                 worker_program_offset: int = 0x0F000000,
                 row_align: int = 64,
                 col_align: int = COL_ALIGN,
                 split_mode: str = "even",
                 allow_unaligned_rows: bool = False,
                 barrier_margin_nops: int = 32,
                 allow_more_than_two_engines: bool = False,
                 workers: Optional[list] = None,
                 worker_map: str = "legacy",
                 worker_arena: Optional[tuple] = None,
                 arena: Optional[PrivateArena] = None,
                 handshake: str = "nops",
                 region_rendezvous: str = "symmetric",
                 verbose: bool = False):
        assert num_engines >= 1, f"num_engines must be >= 1, got {num_engines}"
        # ENGINE-COUNT GATE. This used to be an unconditional opt-in ("the device
        # caps at 2 engines"). Multi-core bitstreams now REPORT their core count
        # through HW_INFO, and 8-engine operation is hardware-verified, so the
        # authority is the bitstream: if it says it has the cores, N is allowed.
        # ``allow_more_than_two_engines`` is kept -- every existing caller passes
        # it -- but is now only needed when HW_INFO has not been probed.
        hw_cores = getattr(user_dma_core, "ANDROMEDA_CORE_COUNT", None)
        if hw_cores is not None:
            assert num_engines <= hw_cores, (
                f"num_engines={num_engines} exceeds the {hw_cores} core(s) HW_INFO "
                f"reports for the loaded bitstream")
        elif num_engines > 2:
            assert allow_more_than_two_engines, (
                f"num_engines={num_engines} > 2 and HW_INFO has not been read, so the "
                f"core count is unknown; pass allow_more_than_two_engines=True to opt in"
            )
        self.primary = primary_ue
        self.num_engines = num_engines
        self.row_align = row_align
        # Column shards are ALWAYS a multiple of this; there is no escape hatch
        # (unlike allow_unaligned_rows). B row blocks, the quantization scale
        # blob and the 128 B SRAM row all agree on 64 elements, and nothing in
        # the matvec pipeline consumes a partial vector.
        assert col_align % COL_ALIGN == 0 and col_align > 0, (
            f"col_align={col_align} must be a positive multiple of "
            f"UE_VECTOR_SIZE={COL_ALIGN}")
        self.col_align = col_align
        self.split_mode = split_mode
        self.allow_unaligned_rows = allow_unaligned_rows
        self.barrier_margin_nops = barrier_margin_nops

        # RENDEZVOUS MECHANISM for the symmetric barrier():
        #   "nops"       -- SET / CHECK(others) / N NOPs / CLEAR. The original, and
        #                   still the default so existing programs keep the exact
        #                   instruction stream they are validated against.
        #   "four_phase" -- SET / CHECK_SET(others) / CLEAR / CHECK_CLEAR(others).
        #                   Same symmetric contract, no timing margin. Strictly more
        #                   robust; ``barrier_margin_nops`` is ignored in this mode.
        # Independent of release()/join(), which are always four-phase.
        assert handshake in ("nops", "four_phase"), \
            f"handshake must be 'nops' or 'four_phase', got {handshake!r}"
        self.handshake = handshake

        # HOW A REGION SYNCHRONISES. The region API (sharded_region,
        # col_sharded_region, k_sharded_region, head_sharded_region) is unchanged
        # in shape either way -- one body replayed per engine -- but the
        # rendezvous around it has two implementations:
        #
        #   "symmetric"    (default, and what every existing caller gets): an
        #       all-to-all barrier at region entry and again at exit. Each engine
        #       waits on all N-1 others, twice per region: O(N^2) checks, and the
        #       re-arm rests on a NOP margin unless handshake="four_phase".
        #
        #   "master_worker": ONE four-phase round WRAPPED AROUND the body --
        #       master SET at entry, join at exit; workers wait at entry, signal at
        #       exit. O(N) checks on the master and exactly one on each worker,
        #       half as many rendezvous, and every flag edge is acknowledged, so it
        #       needs no timing margin at any engine count.
        #
        # The default is "symmetric" ON PURPOSE: it emits the identical
        # instruction stream those models are validated against, so opting in is
        # a per-model decision, not a library-wide change.
        assert region_rendezvous in ("symmetric", "master_worker"), (
            f"region_rendezvous must be 'symmetric' or 'master_worker', "
            f"got {region_rendezvous!r}")
        self.region_rendezvous = region_rendezvous
        self._mw_round_open = False
        # Latched by the first rendezvous emitted; see _latch_rendezvous().
        self._rendezvous_mode: Optional[str] = None
        self.verbose = verbose

        # PRIVATE PER-ENGINE MAP (opt-in). Three ways in:
        #   worker_map="legacy"       -- no private map; the original worker bases.
        #   worker_map="private_low"  -- the default low-2 GB arena, below the model
        #                                map at DRAM_START_ADDR.
        #   worker_map="private" + worker_arena=(base, size), or arena=<PrivateArena>
        #                             -- an explicit arena, for a model that has put
        #                                its OWN map in the low 2 GB.
        # Regions cover EVERY engine including the primary: engine 0 keeps its own
        # model map for activations, but its WEIGHT SHARD lives in region 0 like
        # everybody else's, so all N shards are laid out by one rule.
        #
        # PASS ``arena`` WHEN THE MODEL ALREADY OWNS ONE. Several schedulers over one
        # address range with one allocator each is how a worker program ends up on
        # another engine's scratch; sharing the arena object keeps a single cursor per
        # engine for the whole run.
        assert worker_map in ("legacy", "private_low", "private"), \
            f"worker_map must be 'legacy', 'private_low' or 'private', got {worker_map!r}"
        if arena is not None:
            assert arena.num_engines >= num_engines, (
                f"shared arena covers {arena.num_engines} engine(s), need {num_engines}")
            worker_map = "private"
        elif worker_map != "legacy":
            a_base, a_bytes = worker_arena if worker_arena is not None else (0, None)
            arena = PrivateArena(num_engines, arena_base=a_base, arena_bytes=a_bytes)
        self.worker_map = worker_map
        self.arena: Optional[PrivateArena] = arena
        self.regions: Optional[list] = arena.regions if arena is not None else None
        if arena is not None and verbose:
            print(arena.describe())

        # Worker engines. Their allocator bases are OWNED HERE: DRAM is one flat
        # space, so independent Python allocators would silently collide with
        # the model's params/tensor/program cursors.
        if worker_dram_base is None and worker_map == "legacy":
            worker_dram_base = user_dma_core.DRAM_START_ADDR + worker_dram_stride
        self.workers: list[UnifiedEngine] = []
        # CONSTRUCTING AN ENGINE DESTROYS THE FIRST 16 KB OF DRAM.
        # UnifiedEngine.init_unified_engine() runs a DRAM read/write self-test
        # that dma_writes 8192 random uint16 to the HARDCODED DRAM_START_ADDR
        # (user_dma_core.py:827-837) -- it does not respect params_dram_base, so
        # every engine ever built stomps the same region. The primary does it
        # too, but harmlessly: that happens before the model uploads anything.
        # The workers are built HERE, which in any real flow is AFTER the model
        # has already written its weights -- and a model whose params allocator
        # starts at DRAM_START_ADDR (the default) has its first rows sitting
        # exactly there. That silently truncated the first N-chunk of an IF4
        # weight blob and cost a debugging session; it is not visible as a NaN,
        # only as one bad column block.
        # We cannot fix init_unified_engine(), so we make it a no-op: snapshot
        # the region, build the workers, put it back. Idempotent, and it makes
        # scheduler-construction order irrelevant.
        # ``workers`` lets the CALLER own the worker engines and share them across
        # schedulers. This is not an optimization -- it is a correctness fix.
        #
        # Each worker's DRAM allocators live IN the UnifiedEngine object. Build a
        # second scheduler over the same arena and its fresh objects restart those
        # allocators at offset 0, so stage 2's worker programs are written on top of
        # stage 1's, byte for byte, at the same addresses. Stage 1 has already
        # cached those addresses, so its next execution launches stage 2's code on
        # the workers: different program, different barrier count, instant desync,
        # and the engines spin on FLAG_CHECK forever (which has no timeout). The
        # host sees a hang, kills the run, and every later process inherits engines
        # that are still busy.
        #
        # It only bites when stages disagree on the engine count -- with a single
        # --engines N the scheduler is reused and there is one allocator. Per-stage
        # counts (--vis_4 --pref_8) are exactly the case that breaks it. Passing a
        # shared pool keeps ONE allocator per engine for the whole run, so each
        # stage's programs land AFTER the previous stage's.
        if workers is not None:
            assert len(workers) >= num_engines - 1, (
                f"shared worker pool has {len(workers)} engine(s), need "
                f"{num_engines - 1} for num_engines={num_engines}")
            self.workers = list(workers[: num_engines - 1])
        else:
            guard = self._save_dram_selftest_region() if num_engines > 1 else None
            for i in range(1, num_engines):
                if self.regions is not None:
                    # private_low: weights at the window base, then the ISA slice,
                    # then tensor scratch -- all below DRAM_START_ADDR, so none of
                    # it can alias the model's own map however the model allocates.
                    r = self.regions[i]
                    p_base, t_base, g_base = r.weight_base, r.tensor_base, r.isa_base
                else:
                    p_base = worker_dram_base + (i - 1) * worker_dram_stride
                    assert p_base != getattr(primary_ue, "_params_dram_base", None), \
                        "worker DRAM base collides with the primary's params base"
                    t_base = p_base + worker_tensor_offset
                    g_base = p_base + worker_program_offset
                self.workers.append(UnifiedEngine(
                    BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * engine_base_stride,
                    params_dram_base=p_base,
                    tensor_dram_base=t_base,
                    program_dram_base=g_base,
                ))
            if guard is not None:
                self._restore_dram_selftest_region(guard)
        self.engines: list[UnifiedEngine] = [primary_ue] + self.workers

        self._per_engine: dict[str, list[int]] = {}
        self._m_regs: dict[tuple[int, int], int] = {}   # (engine_idx, rows) -> gpr
        self._n_regs: dict[tuple[int, int], int] = {}   # (engine_idx, cols) -> gpr
        self._col_outputs: dict[str, list[int]] = {}
        self._in_region = False
        self._region_count = 0
        self._program_open = False
        self._worker_prog_addrs: list[int] = []

        # --- master/worker sharding state (all unused by the region API) ---
        # Allocation lives in self.arena (a PrivateArena), never in the scheduler:
        # see the note above about one cursor per engine per run.
        self._weights: dict[str, ShardedWeight] = {}
        self._persistent_prog: dict[int, int] = {}        # engine -> body address
        self._persistent_preamble: dict[int, int] = {}    # engine -> preamble address
        self._persistent_body_word: dict[int, int] = {}   # engine -> body word address
        self._persistent_aligned_reg: dict[int, Optional[int]] = {}

    # ------------------------------------------------- engine-construction --
    def _save_dram_selftest_region(self):
        """Read back the region ``init_unified_engine()`` is about to destroy.

        See the comment at the worker-construction loop. The self-test writes
        8192 uint16 at ``DRAM_START_ADDR``; we guard twice that so a change to
        its size cannot re-open the hole silently.
        """
        import torch
        # MUST be bfloat16, not uint8. ``dma_read`` only round-trips bits
        # losslessly for bf16 (and int32 registers): every other dtype goes
        # through ``tensor_uint16.to(buffer.dtype)``, which is a NUMERIC CAST,
        # not a reinterpretation (user_dma_core.py:1100). A uint8 buffer
        # silently keeps the low byte of each 16-bit word and half the length,
        # so "restoring" it corrupts the region worse than the self-test did.
        buf = torch.zeros(DRAM_SELFTEST_GUARD_BYTES // 2, dtype=torch.bfloat16)
        self.primary.dma_read(self.primary.c2h_device,
                              user_dma_core.DRAM_START_ADDR, buf,
                              DRAM_SELFTEST_GUARD_BYTES)
        return buf

    def _restore_dram_selftest_region(self, buf) -> None:
        self.primary.dma_write(self.primary.h2c_device,
                               user_dma_core.DRAM_START_ADDR, buf,
                               DRAM_SELFTEST_GUARD_BYTES)

    # ------------------------------------------------------------------ rows
    def split_rows(self, M: int, mode: Optional[str] = None) -> list[tuple[int, int]]:
        """Return [(row_offset, row_count)] per engine for a full row count M.

        ``mode`` selects the partitioning rule (default ``self.split_mode``):

        ``"even"``   split M as evenly as possible, then ASSERT every shard is
                     ``row_align``-aligned. Right when M/n happens to land on the
                     alignment (the vision encoder: 256/2 = 128).
        ``"blocks"`` split by whole ``row_align``-row BLOCKS, deliberately
                     UNEVEN so every shard stays aligned. This is the only
                     workable rule for the pi05 prefix, whose real sequence
                     lengths are 320/576/832: an even split gives 160/288/416,
                     none of which is 64-aligned, so ``"even"`` would assert on
                     EVERY real prefix. Block-splitting 832 -> 448/384 costs a
                     little load imbalance and keeps every kernel's 64-row
                     assumption intact -- strictly better than relaxing the
                     alignment via ``allow_unaligned_rows``.
        """
        n = self.num_engines
        mode = mode or self.split_mode
        if mode == "blocks" and n > 1:
            a = self.row_align
            assert M % a == 0, (
                f"split_rows(mode='blocks'): M={M} is not a multiple of row_align={a}")
            blocks = M // a
            assert blocks >= n, (
                f"split_rows(mode='blocks'): M={M} is only {blocks} block(s) of {a} "
                f"rows, too few for num_engines={n}")
            base, rem = divmod(blocks, n)
            counts = [a * (base + (1 if i < rem else 0)) for i in range(n)]
            offsets = [sum(counts[:i]) for i in range(n)]
            return list(zip(offsets, counts))
        assert mode in ("even", "blocks"), f"unknown split mode {mode!r}"
        base, rem = divmod(M, n)
        counts = [base + (1 if i < rem else 0) for i in range(n)]
        assert all(c >= 1 for c in counts), f"num_engines={n} too large for M={M}"
        if not self.allow_unaligned_rows and n > 1:
            for c in counts:
                assert c % self.row_align == 0, (
                    f"shard row count {c} is not {self.row_align}-aligned (M={M}, "
                    f"num_engines={n}); kernels assume {self.row_align}-aligned row "
                    f"blocks. Pass allow_unaligned_rows=True only if you have "
                    f"verified the ops involved."
                )
        offsets = [sum(counts[:i]) for i in range(n)]
        return list(zip(offsets, counts))

    # ------------------------------------------------------------- columns --
    def split_cols(self, N: int, remainder: str = "leading",
                   max_engines: Optional[int] = None) -> list[tuple[int, int]]:
        """Return [(col_offset, col_count)] per engine for a full column count N.

        ``max_engines`` caps how many engines take part, so a weight too narrow
        to give every engine a 64-column block can still be shared by the few it
        does fill (gemma3's N=256 K projection: 4 blocks, 4 engines, the rest
        idle for that op). The returned list is then SHORTER than num_engines --
        engine i owns entry i, and engines past the end own nothing.

        ``remainder`` decides WHICH engines carry the extra 64-column blocks when N
        does not divide evenly:

          ``"leading"``  (default) -- engines 0..rem-1 get one block more. This is
              the original behaviour and it is the DEFAULT ON PURPOSE: callers mirror
              this split host-side to pre-slice weights before upload, and changing it
              under them would slice the weights one way while emitting the matmul the
              other -- silent garbage, not a crash.
          ``"trailing"`` -- the HIGHEST-numbered engines get the extra blocks, so
              engine 0 gets the smallest shard. Right for the master/worker topology,
              where engine 0 also runs everything that is NOT sharded (attention, the
              norms, the residuals) and every round ends with the master waiting on the
              slowest worker: giving the master less matmul shortens that wait.

        Column shards are split by whole ``col_align``-element blocks and are
        deliberately allowed to be UNEVEN (the trailing engines get one block
        less) rather than ever unaligned -- the same trade ``split_rows``
        makes in ``"blocks"`` mode, and here it is the ONLY rule: a partial
        64-element vector is not something the matvec pipeline, the B row
        block, or the scale blob can express.

        At bpe=2 a ``col_align``-element block is exactly 128 bytes, so every
        offset this produces is a whole SRAM row (asserted in :func:`_shifted`).
        """
        n = self.num_engines if max_engines is None else min(self.num_engines,
                                                             max(1, max_engines))
        a = self.col_align
        if n == 1:
            return [(0, N)]
        assert N % a == 0, (
            f"split_cols: N={N} is not a multiple of col_align={a}. Column shards "
            f"must be multiples of {a} elements; pad N or shard a different axis.")
        blocks = N // a
        assert blocks >= n, (
            f"split_cols: N={N} is only {blocks} block(s) of {a} columns, too few "
            f"for {n} engine(s); pass max_engines<={blocks} to shard it over "
            f"a subset and leave the rest idle for this op")
        assert remainder in ("leading", "trailing"), \
            f"remainder must be 'leading' or 'trailing', got {remainder!r}"
        base, rem = divmod(blocks, n)
        if remainder == "leading":
            counts = [a * (base + (1 if i < rem else 0)) for i in range(n)]
        else:
            counts = [a * (base + (1 if i >= n - rem else 0)) for i in range(n)]
        offsets = [sum(counts[:i]) for i in range(n)]
        return list(zip(offsets, counts))

    def split_k(self, K: int) -> list[tuple[int, int]]:
        """Return [(k_offset, k_count)] per engine for a REDUCTION split.

        Use when the fat dimension is K, not N (pi05's ``mlp down``,
        4096 -> 1024): each engine multiplies its K-slice and produces a FULL
        [M, N] PARTIAL SUM, which :meth:`reduce_add` then combines.

        THE SLICED WEIGHT MUST BE PREPARED ON THE HOST. ``B`` is N x K
        row-major so ``B[:, k0:k0+Kc]`` is strided (one gap per row) and cannot
        be reached by shifting ``B_DRAM_ADDR``; upload
        ``B[:, k0:k0+Kc].contiguous()`` as its own blob per engine (and, for
        IF4/IF8, quantize that slice -- the scale blocking is over the sliced
        K, not the original). This method only hands you the ranges.
        """
        return self.split_cols(K)

    # --------------------------------------------------------------- heads --
    def split_heads(self, H: int, gqa_ratio: int = 1,
                    mode: str = "groups") -> list[tuple[int, int]]:
        """Return [(head_offset, head_count)] per engine for H attention heads.

        ``mode="groups"`` (default) splits on KV-GROUP boundaries, so every
        engine owns whole groups and each KV plane is transposed to ``V^T``
        exactly once across the machine. Parallelism caps at ``H // gqa_ratio``.

        ``mode="qheads"`` splits on Q-HEAD boundaries, letting a group straddle
        engines. This is CORRECT, not a relaxation of a correctness rule: the
        constraint the kernel imposes is per CALL (one call re-reads
        ``K_DRAM_ADDR`` unshifted and builds ``V^T`` once, so its heads must
        share a KV head), and an engine owning a partial group simply issues
        more, smaller calls -- see :meth:`HeadShardContext.call_runs`. K/V are
        read-only, so two engines reading one KV plane never conflict.

        The cost is a duplicated ``V^T`` per straddled group: a
        [head_dim, seq_len] transpose against a head whose real work is two
        seq_len**2 * head_dim matmuls, i.e. sub-1% at typical shapes. Use
        ``"qheads"`` when ``H // gqa_ratio < num_engines`` would otherwise
        refuse to shard at all (heavily-GQA models: 16 Q heads over 2 KV heads
        gives only 2 groups), and ``"groups"`` -- the tidier default -- when
        there are enough groups to go around.
        """
        n = self.num_engines
        assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
        assert H % gqa_ratio == 0, (
            f"split_heads: H={H} is not a multiple of gqa_ratio={gqa_ratio}")
        assert mode in ("groups", "qheads"), f"unknown head split mode {mode!r}"
        if n == 1:
            return [(0, H)]

        if mode == "qheads":
            assert H >= n, (
                f"split_heads(mode='qheads'): H={H} head(s) is fewer than "
                f"num_engines={n}; there is nothing left to split.")
            base, rem = divmod(H, n)
            counts = [base + (1 if i < rem else 0) for i in range(n)]
            offsets = [sum(counts[:i]) for i in range(n)]
            return list(zip(offsets, counts))

        groups = H // gqa_ratio
        assert groups >= n, (
            f"split_heads: H={H} is only {groups} KV group(s) of {gqa_ratio} "
            f"head(s), too few for num_engines={n}. Pass mode='qheads' to split "
            f"within groups (costs a duplicated V^T per straddled group), shard "
            f"a different axis, or run attention on the primary alone.")
        base, rem = divmod(groups, n)
        counts = [gqa_ratio * (base + (1 if i < rem else 0)) for i in range(n)]
        offsets = [sum(counts[:i]) for i in range(n)]
        return list(zip(offsets, counts))

    @staticmethod
    def attn_scratch_bytes(seq_len: int, head_dim: int, elem_bytes: int = 2) -> int:
        """Per-engine flash scratch: ``SCRATCH_VT`` then ``SCRATCH_SM``.

        ``V^T`` is [head_dim, seq_len] and the attention-probability matrix is
        [seq_len, seq_len] (``nn_lib.py:1024``), so this grows as seq_len**2 and
        is the binding constraint on head sharding -- NOT program size. At
        seq_len 2048 it is ~8 MB per engine; at 4096, ~32 MB. Budget it against
        the 4 GB map before committing to an engine count.
        """
        return (head_dim + seq_len) * seq_len * elem_bytes

    def alloc_attn_scratch(self, name: str, seq_len: int, head_dim: int,
                           primary_addr: int, elem_bytes: int = 2) -> None:
        """Register the per-engine flash scratch for a head-sharded region.

        ``primary_addr`` is the scratch the MODEL already owns (engine 0 keeps
        using it); the workers get their own from the scheduler allocator. Thin
        wrapper over :meth:`register_per_engine` that computes the size, so a
        caller cannot under-allocate and have engine 1 write into whatever
        follows.
        """
        self.register_per_engine(
            name, primary_addr,
            self.attn_scratch_bytes(seq_len, head_dim, elem_bytes))

    def begin_head_sharded(self, H: int, seq_len: int, head_dim: int,
                           gqa_ratio: int = 1,
                           elem_bytes: int = 2,
                           mode: str = "groups") -> list[HeadShardContext]:
        """Rendezvous, then open a head-sharded region, one context per engine."""
        assert self._program_open, "begin_head_sharded() without begin_program()"
        assert not self._in_region, "nested sharded regions are not supported"
        split = self.split_heads(H, gqa_ratio, mode=mode)
        self._region_enter()
        self._in_region = True
        self._region_count += 1
        return [HeadShardContext(self, i, ue, H, split[i][0], split[i][1],
                                 gqa_ratio, seq_len, head_dim, elem_bytes)
                for i, ue in enumerate(self.engines)]

    def head_sharded_region(self, H: int, seq_len: int, head_dim: int,
                            body: Callable[[HeadShardContext], None],
                            gqa_ratio: int = 1, elem_bytes: int = 2,
                            mode: str = "groups", join: bool = True) -> None:
        """Replay ``body(ctx)`` once per engine over a head split.

        ``join`` defaults to True and should almost always stay there: the next
        thing after attention is o_proj, which reads the CONCATENATED
        [seq_len, H*head_dim] across every engine's slice -- the textbook
        cross-shard read the exit barrier exists for.
        """
        contexts = self.begin_head_sharded(H, seq_len, head_dim, gqa_ratio,
                                           elem_bytes, mode=mode)
        for ctx in contexts:
            body(ctx)
        self.end_sharded(join=join)

    def head_sharded_attention(self, H: int, seq_len: int, head_dim: int,
                               Q_addr: int, K_addr: int, V_addr: int,
                               OUT_addr: int, IDENTITY_addr: int,
                               *,
                               gqa_ratio: int = 1,
                               bias_addr: Optional[int] = None,
                               bias_per_head: bool = False,
                               scratch_name: str = "attn_scratch",
                               mode: str = "groups",
                               kernel: Optional[Callable] = None,
                               elem_bytes: int = 2,
                               join: bool = True) -> None:
        """Head-sharded prefill attention -- the whole stage in one call.

        ``kernel`` defaults to ``nn_lib.prefill_flash_attention_core``; pass a
        different one (the decode variant, a PBI batched flash) with the same
        keyword signature. Register the scratch with :meth:`alloc_attn_scratch`
        first.

        Per ``project_pbi_flash_back_to_back_bug``, flash stays on the LEGACY
        (non-PBI) path -- head sharding is static per engine, so each engine
        carries its own full unroll and the total program bytes are merely
        redistributed, not multiplied. Check :meth:`worker_program_bytes`.
        """
        if kernel is None:
            import nn_lib
            kernel = nn_lib.prefill_flash_attention_core

        # head_dim 72 (MoonViT) fails the 128 B assertion inside _shifted; pad
        # to 128 first, exactly as the single-engine path already must.
        assert (seq_len * head_dim * elem_bytes) % SRAM_ROW_BYTES == 0, (
            f"head plane {seq_len}x{head_dim} is not a whole number of "
            f"{SRAM_ROW_BYTES} B SRAM rows; pad head_dim before sharding")

        def body(ctx: HeadShardContext) -> None:
            # ONE KERNEL CALL PER KV GROUP, num_q_heads=gqa_ratio.
            #
            # ``num_q_heads`` is NOT "how many heads to process" -- it is "how
            # many Q heads share this ONE K/V head". Inside the kernel only
            # ``q_base``/``out_base`` advance with the head index; ``K_DRAM_ADDR``
            # is re-read unshifted every iteration and ``SCRATCH_VT`` is built
            # ONCE before the loop (nn_lib.py:1054, :1110). Passing the engine's
            # full head count therefore computes every head against KV head 0 --
            # finite, plausible, and wrong (-2.4 dB, caught by the num_engines=1
            # passthrough case).
            #
            # So the engine iterates its own KV groups and hands each call the
            # gqa_ratio Q heads that genuinely share that group. With
            # gqa_ratio == 1 (MHA) this is one call per head, which is simply
            # what the kernel's batching means for MHA -- there is nothing to
            # batch when no two Q heads share a K/V.
            # call_runs() yields maximal spans of Q heads sharing one KV head --
            # one group per run when the engine owns whole groups, ragged short
            # runs at the ends when a group straddles engines (mode="qheads").
            for q_head, kv_head, n_q in ctx.call_runs():
                kernel(ctx.ue, head_dim, seq_len,
                       Q_DRAM_ADDR=ctx.q_addr(Q_addr, head=q_head),
                       K_DRAM_ADDR=ctx.kv_addr(K_addr, kv_head=kv_head),
                       V_DRAM_ADDR=ctx.kv_addr(V_addr, kv_head=kv_head),
                       OUTPUT_DRAM_ADDR=ctx.out_addr(OUT_addr, head=q_head),
                       SCRATCH_DRAM_ADDR=ctx.scratch(scratch_name),
                       IDENTITY_DRAM_ADDR=IDENTITY_addr,   # shared, read-only
                       BIAS_DRAM_ADDR=ctx.bias_addr(bias_addr,
                                                    per_head=bias_per_head,
                                                    head=q_head),
                       num_q_heads=n_q)

        self.head_sharded_region(H, seq_len, head_dim, body,
                                 gqa_ratio=gqa_ratio, elem_bytes=elem_bytes,
                                 mode=mode, join=join)

    def alloc_col_output(self, name: str, M: int, N: int, elem_bytes: int = 2) -> list[int]:
        """Allocate one contiguous ``[M, cols]`` output buffer PER ENGINE.

        This is the answer to "the output column block is strided": it is not,
        because ``matmat_mul_core`` writes back with stride ``N*bpe`` for the
        ``N`` IT WAS GIVEN (``user_dma_core.py:5578``), so calling it with
        ``N=cols`` produces a DENSE ``[M, cols]`` block. Each engine writes its
        own, nothing overlaps, and no strided writeback is needed -- which is
        just as well, since the dynamic/PBI path does not implement one
        (``user_dma_core.py:5688``).

        Buffers are 128-byte aligned so every downstream row offset stays on an
        SRAM row. Returns the per-engine address list, primary first.
        """
        assert name not in self._col_outputs, f"column output {name!r} already allocated"
        split = self.split_cols(N)
        addrs = []
        for (off, cols), ue in zip(split, self.engines):
            a = ue.allocate_tensor_dram(M * cols * elem_bytes,
                                        label=f"col_out_{name}_{off}",
                                        align_bytes=SRAM_ROW_BYTES)
            assert a % SRAM_ROW_BYTES == 0, \
                f"column output {name!r} base 0x{a:x} is not {SRAM_ROW_BYTES} B aligned"
            addrs.append(a)
        self._col_outputs[name] = addrs
        return list(addrs)

    def col_output_addr(self, name: str, engine_idx: int) -> int:
        assert name in self._col_outputs, (
            f"unknown column output {name!r}; allocate it with alloc_col_output() "
            f"before the column-sharded region")
        return self._col_outputs[name][engine_idx]

    def col_output_addrs(self, name: str) -> list[int]:
        return list(self._col_outputs[name])

    def begin_col_sharded(self, N: int) -> list[ColumnShardContext]:
        """Rendezvous, then open an N-sharded region, one context per engine."""
        assert self._program_open, "begin_col_sharded() without begin_program()"
        assert not self._in_region, "nested sharded regions are not supported"
        split = self.split_cols(N)
        self._region_enter()
        self._in_region = True
        self._region_count += 1
        self._n_regs.clear()   # column counts are per-region
        return [ColumnShardContext(self, i, ue, N, split[i][0], split[i][1])
                for i, ue in enumerate(self.engines)]

    def col_sharded_region(self, N: int, body: Callable[[ColumnShardContext], None],
                           join: bool = True) -> None:
        """Replay ``body(ctx)`` once per engine inside one barrier-free region.

        Pass ``join=False`` when the NEXT region stays in the same column lane
        (gate -> up -> gelu-multiply -> ...): elementwise consumers of a column
        shard need no cross-engine data at all, and that is the whole point of
        the mode. Only join before something that reads columns another engine
        owns (a K-split reduction, an attention, a full-N gather).
        """
        contexts = self.begin_col_sharded(N)
        for ctx in contexts:
            body(ctx)
        self.end_sharded(join=join)

    def k_sharded_region(self, K: int, body: Callable[[KShardContext], None],
                         join: bool = False) -> None:
        """Replay ``body(ctx)`` once per engine over a K (reduction) split.

        Defaults to ``join=False`` because the caller's very next statement
        should be :meth:`reduce_add`, which opens with its own barrier -- the
        partials are meaningless until it runs, so the join belongs there.
        """
        assert self._program_open, "k_sharded_region() without begin_program()"
        assert not self._in_region, "nested sharded regions are not supported"
        split = self.split_k(K)
        self._region_enter()
        self._in_region = True
        self._region_count += 1
        self._n_regs.clear()
        for i, ue in enumerate(self.engines):
            body(KShardContext(self, i, ue, K, split[i][0], split[i][1]))
        self.end_sharded(join=join)

    def reduce_add(self, partial_addrs: list[int], out_addr: int, M: int, N: int,
                   join: bool = True) -> None:
        """Cross-engine SUM of per-engine partial [M, N] results (K-split join).

        Deliberately explicit and deliberately named: this is the ONLY place in
        the project where engines' results are combined arithmetically rather
        than merely concatenated by address, and hiding it inside a region
        helper would make a genuinely expensive step invisible.

        Emits: barrier -> (n-1) ``eltwise_core_dram`` ELTWISE_ADD of M*N bf16
        ON THE PRIMARY, workers idle -> barrier (if ``join``). The adds are
        serialized against everything else; see the cost note in the module
        docstring. Splitting the add back across engines is not done here
        because the shapes this exists for have M=64, a single row block that
        cannot be row-split at all.

        ``partial_addrs[0]`` may alias ``out_addr`` (accumulate in place).
        """
        assert not self._in_region, "reduce_add() inside an open sharded region"
        assert len(partial_addrs) == self.num_engines, (
            f"reduce_add: {len(partial_addrs)} partial(s) for "
            f"{self.num_engines} engine(s)")
        for a in partial_addrs + [out_addr]:
            assert a % AXI_BEAT_BYTES == 0, \
                f"reduce_add: address 0x{a:x} is not {AXI_BEAT_BYTES} B AXI-beat aligned"
        if self.num_engines == 1:
            # Exact passthrough: a one-engine "reduction" is the identity, and
            # emitting a copy would break byte-identity for no arithmetic.
            assert partial_addrs[0] == out_addr, (
                "reduce_add with num_engines=1 is the identity; point the single "
                "partial at out_addr instead of asking for a copy")
            return
        self.barrier()
        acc = partial_addrs[0]
        for src in partial_addrs[1:]:
            self.primary.eltwise_core_dram(
                M=M, N=N, dram_a=acc, dram_b=src, dram_out=out_addr,
                mode=user_dma_core.UE_MODE.ELTWISE_ADD,
            )
            acc = out_addr
        if join:
            self.barrier()

    # -------------------------------------------------------- PER_ENGINE ---
    def register_per_engine(self, name: str, primary_addr: int, size_bytes: int,
                            init_tensor=None) -> None:
        """Duplicate a scratch buffer that is WRITTEN as scratch.

        The primary keeps its existing model address; each worker gets its own
        copy from the scheduler-owned worker allocator. ``init_tensor`` (if
        given) is uploaded to the worker copies only -- the primary's copy is
        already initialized by the model.
        """
        assert name not in self._per_engine, f"per-engine buffer {name!r} already registered"
        addrs = [primary_addr]
        for w in self.workers:
            a = w.allocate_tensor_dram(size_bytes, label=f"per_engine_{name}")
            if init_tensor is not None:
                w.dma_to_accelerator_memory(a, init_tensor)
            addrs.append(a)
        self._per_engine[name] = addrs

    def register_per_engine_addrs(self, name: str, addrs: list[int]) -> None:
        """Register caller-allocated private scratch addresses.

        Use this when a model owns one shared DRAM allocator and the worker's
        scheduler allocator is reserved for ISA. Addresses must be distinct and
        supplied in engine order (primary first).
        """
        assert len(addrs) == self.num_engines, (
            f"per-engine buffer {name!r} needs {self.num_engines} addresses, "
            f"got {len(addrs)}")
        assert len(set(addrs)) == len(addrs), (
            f"per-engine buffer {name!r} addresses must be distinct")
        for addr in addrs:
            assert addr % AXI_BEAT_BYTES == 0, (
                f"per-engine buffer {name!r} address 0x{addr:x} is not "
                f"{AXI_BEAT_BYTES} B AXI-beat aligned")
        normalized = list(addrs)
        if name in self._per_engine:
            assert self._per_engine[name] == normalized, (
                f"per-engine buffer {name!r} already registered with different addresses")
            return
        self._per_engine[name] = normalized

    def per_engine_addr(self, name: str, engine_idx: int) -> int:
        assert name in self._per_engine, (
            f"unknown PER_ENGINE buffer {name!r}; register it with "
            f"register_per_engine() before the sharded region"
        )
        return self._per_engine[name][engine_idx]

    def refresh_per_engine(self, name: str, tensor) -> None:
        """Re-upload the initial contents to EVERY copy (primary included).

        Needed for scratch such as pi05's ``vis_zeros_addr`` that the kernel
        dirties, so a second execution does not start from garbage.
        """
        for ue, addr in zip(self.engines, self._per_engine[name]):
            ue.dma_to_accelerator_memory(addr, tensor)

    # ----------------------------------------------------------- program ---
    def begin_program(self) -> None:
        """Open the worker capture sessions. The primary's capture is the
        model's own and is never opened or closed here."""
        assert not self._program_open, "begin_program() called twice"
        assert self.primary.is_capture_on, (
            "the primary engine must already be capturing: this class emits into "
            "the model's EXISTING capture session"
        )
        for w in self.workers:
            w.clear_capture_buffer()
            w.reset_isa_reg_counter()
            w.reset_inst_ptr_counter()
            w.start_capture()
        # THE RENDEZVOUS LATCH IS PER PROGRAM. What must agree on a topology is
        # the pair of streams that RUN TOGETHER; two programs compiled on one
        # scheduler and launched separately never rendezvous with each other. A
        # model that row-shards its prefill through sharded_region (symmetric) and
        # then master/worker-shards its decode on the same engines is doing
        # nothing wrong, so each begin_program() starts the choice afresh.
        self._rendezvous_mode = None
        self._program_open = True

    def finalize(self) -> list[int]:
        """Halt, close and flush every worker program to DRAM; return the worker
        program addresses.

        The primary's ``stop_capture()`` / program write stays the model's job.

        CALLERS THAT COMPILE MORE THAN ONE STAGE ON ONE SCHEDULER (pi05: encoder
        then prefix) MUST KEEP THE RETURNED LIST: ``self._worker_prog_addrs`` is
        overwritten by each finalize(), so a bare ``start_workers()`` after the
        second stage would relaunch the WRONG worker program for the first.
        Pass the saved list to :meth:`start_workers`.
        """
        assert self._program_open, "finalize() without begin_program()"
        assert not self._in_region, "finalize() inside an open sharded region"
        if self._mw_round_open:
            # The last region closed with join=False. In the symmetric shape that
            # is harmless (no trailing barrier); here it would leave the master's
            # release standing with no join, and every worker waiting on a close
            # that never comes. Shut the round before the HALT.
            self._region_exit(join=True)
        self._worker_prog_addrs = []
        for wi, w in enumerate(self.workers):
            w.generate_instruction_halt()
            w.stop_capture()
            addr = w.get_program_dram_addr()
            try:
                w.write_captured_instructions_to_dram(addr)
            except TypeError:
                # DIAGNOSTIC (temporary): Instructions.get_bytes() has been seen
                # failing with "cannot convert '_struct.Struct' object to
                # bytearray", which means a non-int landed in inst.words. Report
                # WHICH worker/instruction and what the bad word actually is --
                # the bare traceback names neither.
                import struct as _s
                print(f"    [finalize] worker {wi} FAILED flushing "
                      f"{w.capture_count} instructions to 0x{addr:x}")
                print(f"    [finalize] struct.pack is {_s.pack!r} -> "
                      f"{type(_s.pack('<I', 1))}")
                for k, inst in enumerate(w.capture_buffer):
                    bad = [(j, type(x).__name__, repr(x))
                           for j, x in enumerate(inst.words)
                           if not isinstance(x, int)]
                    if bad:
                        print(f"    [finalize] inst #{k} has non-int words: {bad}")
                        # NOT inst!r -- Instructions.__repr__ formats every word
                        # with 0x{w:08X} and would raise on the non-int itself.
                        print(f"    [finalize] inst #{k} words = {inst.words}")
                        break
                else:
                    print("    [finalize] every word is an int -- the failure is "
                          "in struct.pack itself, not the instruction stream")
                raise
            w.allocate_program_dram(w.get_capture_instruction_size_bytes())
            self._worker_prog_addrs.append(addr)
        self._program_open = False
        return list(self._worker_prog_addrs)

    def start_workers(self, prog_addrs: Optional[list[int]] = None,
                      aligned_seq_len: Optional[int] = None,
                      gpr_sets_by_worker: Optional[list[list[tuple[int, int]]]] = None) -> None:
        """Launch every worker program. Call BEFORE launching the primary's
        program, on EVERY execution (each run ends with the workers halted).

        ``prog_addrs`` overrides the most recent finalize()'s addresses -- pass
        the stage's saved list when one scheduler carries several stages.

        ``gpr_sets_by_worker`` primes runtime registers in ordinary worker
        programs (those emitted between :meth:`begin_program` and
        :meth:`finalize`).  It is indexed by worker, not engine, and each entry
        is a list of ``(register, value)`` pairs.  A tiny preamble is rewritten
        immediately after that worker's loaded program and jumps to its normal
        entry point.  This is useful when a worker participates in an otherwise
        unrolled program but one kernel needs a live runtime dimension.

        NEW -- ``aligned_seq_len`` is for workers built by
        :meth:`emit_worker_program` whose rounds contain attention. Their body is
        compiled ONCE but the KV length grows every token, so each worker is entered
        through a freshly written 2-instruction PREAMBLE (``add_set`` into the body's
        aligned-length register, then a jump into the body) -- the same way a model
        primes its own folded decode program. Ignored by workers that have no such
        register.
        """
        if self._persistent_prog and prog_addrs is None:
            # Persistent (master/worker) workers: addresses are per engine index and
            # the entry point may be a per-token preamble rather than the body itself.
            for idx in self.worker_indices():
                ue = self.engines[idx]
                reg = self._persistent_aligned_reg.get(idx)
                if aligned_seq_len is None or reg is None:
                    ue.start_execute_from_dram(self._persistent_prog[idx])
                    continue
                ue.clear_inst_id()
                ue.start_capture()
                ue.generate_instruction_add_set(reg, aligned_seq_len)
                ue.generate_instruction_jump_abs(self._persistent_body_word[idx])
                ue.stop_capture()
                ue.write_captured_instructions_to_dram(self._persistent_preamble[idx])
                ue.clear_capture_buffer()
                ue.start_execute_from_dram(self._persistent_preamble[idx])
            return
        addrs = self._worker_prog_addrs if prog_addrs is None else prog_addrs
        assert len(addrs) == len(self.workers), \
            f"start_workers: {len(addrs)} program address(es) for {len(self.workers)} worker(s)"
        if gpr_sets_by_worker is not None and len(gpr_sets_by_worker) != len(self.workers):
            raise ValueError(
                "gpr_sets_by_worker must contain one register list per worker: "
                f"got {len(gpr_sets_by_worker)} for {len(self.workers)} workers")
        preambles = getattr(self, "_runtime_worker_preambles", None)
        if preambles is None:
            preambles = self._runtime_worker_preambles = {}
        for wi, (w, addr) in enumerate(zip(self.workers, addrs)):
            reg_sets = (gpr_sets_by_worker[wi]
                        if gpr_sets_by_worker is not None else [])
            if not reg_sets:
                w.start_execute_from_dram(addr)
                continue
            # Reserve one aligned slot after the loaded image on first use, then
            # rewrite that same slot every launch. This keeps the preamble from
            # consuming ISA space once per decode token and gives the arena its
            # normal hard bounds check.
            preamble_key = (wi, int(addr), tuple(reg for reg, _ in reg_sets))
            preamble_addr = preambles.get(preamble_key)
            if preamble_addr is None:
                preamble_bytes = ((len(reg_sets) + 1)
                                  * user_dma_core.INSTRUCTION_SIZE_BYTES)
                preamble_addr = w.allocate_program_dram(
                    preamble_bytes,
                    label=f"worker{wi + 1}_runtime_preamble")
                self._check_isa_fits(wi + 1, preamble_addr, preamble_bytes)
                preambles[preamble_key] = preamble_addr
            w.clear_inst_id()
            w.start_capture()
            for reg, value in reg_sets:
                w.generate_instruction_add_set(reg, value)
            w.generate_instruction_jump_abs(
                user_dma_core.ue_35bit_addr_shifter(addr))
            w.stop_capture()
            w.write_captured_instructions_to_dram(preamble_addr)
            w.clear_capture_buffer()
            w.start_execute_from_dram(preamble_addr)

    def worker_program_bytes(self) -> int:
        return sum(w.get_capture_instruction_size_bytes() for w in self.workers)

    # ----------------------------------------------------------- regions ---
    def begin_sharded(self, M: int) -> list[ShardContext]:
        """Rendezvous, then open a sharded region, returning one
        :class:`ShardContext` per engine."""
        assert self._program_open, "begin_sharded() without begin_program()"
        assert not self._in_region, "nested sharded regions are not supported"
        split = self.split_rows(M)
        self._region_enter()
        self._in_region = True
        self._region_count += 1
        self._m_regs.clear()   # row counts are per-region
        return [ShardContext(self, i, ue, M, split[i][0], split[i][1])
                for i, ue in enumerate(self.engines)]

    def end_sharded(self, join: bool = True) -> None:
        """Close the region; ``join`` emits the exit rendezvous so the primary
        (and every worker) sees all rows before anything downstream reads them.
        Pass ``join=False`` only if you emit :meth:`barrier` yourself later."""
        assert self._in_region, "end_sharded() without begin_sharded()"
        self._in_region = False
        self._region_exit(join)

    def _region_enter(self) -> None:
        """Rendezvous at region ENTRY, in whichever shape this scheduler uses."""
        if self.region_rendezvous == "symmetric":
            self.barrier()
            return
        if self.num_engines == 1:
            return
        if self._mw_round_open:
            # A previous region closed with join=False, meaning "stay in this
            # lane": the round it opened is still the current one, and opening a
            # second SET inside it would be a protocol error, not a barrier.
            return
        self.release()
        for idx in self.worker_indices():
            self.begin_worker_round(idx)
        self._mw_round_open = True

    def _region_exit(self, join: bool) -> None:
        """Rendezvous at region EXIT, or defer it when the caller passed join=False."""
        if self.region_rendezvous == "symmetric":
            if join:
                self.barrier()
            return
        if self.num_engines == 1 or not join:
            return
        for idx in self.worker_indices():
            self.end_worker_round(idx)
        self.join()
        self._mw_round_open = False

    def _latch_rendezvous(self, kind: str) -> None:
        """Pin this scheduler to ONE rendezvous topology and refuse a mix.

        ``barrier()`` is symmetric -- every engine's stream is emitted in lockstep by
        one region body -- while ``release()``/``join()`` are asymmetric, with the
        workers running their own persistent program. Their instruction sequences are
        not compatible: a master waiting in ``join()`` for a worker that is sitting in
        a symmetric ``barrier()`` waits forever, and FLAG_CHECK has NO TIMEOUT, so the
        failure is an unkillable hang rather than an error. Catching it at emit time is
        the only place it can be caught cheaply.

        Scope: ONE PROGRAM. :meth:`begin_program` clears the latch -- see there.
        """
        if self._rendezvous_mode is None:
            self._rendezvous_mode = kind
            return
        assert self._rendezvous_mode == kind, (
            f"this scheduler already emitted a {self._rendezvous_mode!r} rendezvous and "
            f"cannot also emit {kind!r}. The symmetric barrier() and the asymmetric "
            f"release()/join() pair do not interoperate inside one program -- mixing "
            f"them deadlocks with no timeout. Use one topology per scheduler.")

    def barrier(self) -> None:
        """Symmetric all-engine rendezvous (see module docstring).

        Every engine raises its own flag and spin-waits on every other engine's. How it
        then RE-ARMS depends on ``handshake``, chosen at construction:

          ``"nops"`` (default) -- wait out ``barrier_margin_nops`` NOPs, then CLEAR.
              The original shape, proven to re-arm by ``flag_rendezvous_repeat_test``.
              The margin is a TIMING argument: it must exceed the completion skew
              between engines and stay under the next round's work.

          ``"four_phase"`` -- CLEAR, then CHECK_CLEAR every other engine. No margin,
              because the re-arm is OBSERVED rather than waited out: no engine can
              leave the barrier until every other engine's flag has been seen to fall,
              so no engine can carry a stale 1 into the next round. Same symmetric
              contract, same call, strictly weaker assumptions -- and it does not
              degrade as engines are added, which the margin does.

        Prefer ``release()``/``join()`` for new master/worker code; this stays the
        right primitive for region bodies that emit every engine in lockstep.
        """
        assert not self._in_region, "barrier() inside an open sharded region"
        if self.num_engines == 1:
            return   # exact passthrough: not one extra instruction
        if self.region_rendezvous == "master_worker":
            # A standalone barrier is one complete round with no work inside it:
            # the master opens and closes, every worker waits and acknowledges.
            # Same meeting point, asymmetric implementation -- so a model can opt
            # in without touching its explicit barrier() call sites.
            assert not self._mw_round_open, (
                "barrier() inside an open master/worker round: the region before "
                "it closed with join=False, so the round is still current")
            self.release()
            for idx in self.worker_indices():
                self.begin_worker_round(idx)
                self.end_worker_round(idx)
            self.join()
            return
        self._latch_rendezvous("symmetric")
        for i, ue in enumerate(self.engines):
            ue.generate_instruction_flag_set()
            for j in range(self.num_engines):
                if j != i:
                    ue.generate_instruction_flag_check_set(target_engine_idx=j)
            if self.handshake == "four_phase":
                ue.generate_instruction_flag_clear()
                for j in range(self.num_engines):
                    if j != i:
                        ue.generate_instruction_flag_check_clear(target_engine_idx=j)
            else:
                for _ in range(self.barrier_margin_nops):
                    ue.generate_instruction_nop()
                ue.generate_instruction_flag_clear()

    # ======================================================================
    # NEW: asymmetric master/worker rendezvous -- RECOMMENDED (see docstring)
    # ======================================================================
    def release(self) -> None:
        """MASTER, phase 1 of a round: raise the release flag and let the workers run.

        Pair with :meth:`join`; between them the master emits its OWN share of the
        round's work. The workers' side of the same round is emitted by
        :meth:`emit_worker_program`, which is what makes this asymmetric: the workers
        are not replaying the master's stream, they are running their own program and
        meeting it once per round.

            master:  SET | work | CHECK_SET(all W) | CLEAR | CHECK_CLEAR(all W)
            worker:  CHECK_SET(0) | work | SET     | CHECK_CLEAR(0) | CLEAR

        Every flag EDGE is acknowledged before the next edge in that direction, so no
        participant can act on a level left over from the previous round. That is a
        correctness argument rather than a timing margin: it needs no delay, and it
        holds however uneven the shards are and however little work the master has
        between rounds -- both of which were load-bearing, and both of which broke, in
        the margin-based scheme this replaces (see the module docstring).

        A no-op at ``num_engines == 1``: not one extra instruction, so a single-core
        build emits the identical stream it always did.
        """
        if self.num_engines == 1:
            return
        self._latch_rendezvous("master_worker")
        self.primary.generate_instruction_flag_set()                                   # 1

    def join(self) -> None:
        """MASTER, phases 2-4: wait for every worker done, close the round, wait re-armed.

        Phase 4 -- waiting for every worker's flag to FALL -- is the phase the old
        margin-based barrier could not express and the reason it degraded as engines
        were added. It is what guarantees the master's NEXT :meth:`release` cannot be
        satisfied by a flag left standing from this round.
        """
        if self.num_engines == 1:
            return
        self._latch_rendezvous("master_worker")
        for idx in self.worker_indices():
            self.primary.generate_instruction_flag_check_set(target_engine_idx=idx)    # 2
        self.primary.generate_instruction_flag_clear()                                 # 3
        for idx in self.worker_indices():
            self.primary.generate_instruction_flag_check_clear(target_engine_idx=idx)  # 4

    def worker_indices(self) -> list[int]:
        """Engine indices 1..N-1. Engine 0 is the primary and never a worker."""
        return list(range(1, self.num_engines))

    def begin_worker_round(self, engine_idx: int) -> None:
        """WORKER, phase 1: wait for the master's release.

        The worker side of :meth:`release` / :meth:`join`, for a model that emits
        its workers' streams itself (an unrolled decoder, say) rather than through
        :meth:`emit_worker_program`. Emit this, then this engine's share of the
        round, then :meth:`end_worker_round`.

        Every worker must run EVERY round the master emits, even one where it has
        no work: a skipped rendezvous desynchronises the group permanently and the
        master then waits forever on a flag that never rises.
        """
        if self.num_engines == 1:
            return
        self._latch_rendezvous("master_worker")
        self.engines[engine_idx].generate_instruction_flag_check_set(target_engine_idx=0)

    def end_worker_round(self, engine_idx: int) -> None:
        """WORKER, phases 2-4: signal done, wait for the round to close, re-arm."""
        if self.num_engines == 1:
            return
        self._latch_rendezvous("master_worker")
        ue = self.engines[engine_idx]
        ue.generate_instruction_flag_set()                             # 2: done
        ue.generate_instruction_flag_check_clear(target_engine_idx=0)  # 3: closed
        ue.generate_instruction_flag_clear()                           # 4: re-armed

    def preclear_flags(self, timeout_seconds: float = 5.0) -> None:
        """Run a tiny program on every engine that just clears its flag.

        Call once before the first execution: a flag left set by an earlier
        program would make the first CHECK pass spuriously.
        """
        # STALE-BUSY RECOVERY. The FPGA is not reset between processes, so a run
        # that died mid-execution leaves its workers spin-waiting at a FLAG_CHECK
        # (which has no timeout) with queue_busy still asserted. The next process
        # then issues this preclear program to an engine that never accepts it:
        # every worker times out here while the primary -- which the dying process
        # usually did halt -- succeeds. Observed directly: engines 1-3 reading
        # queue_ctrl=0x010E01B0 (busy) while 0 and 4-7 read 0x000E00B0 (idle).
        #
        # A bare SW_RESET register write clears it (verified: busy 1 -> 0 on all
        # three). Deliberately NOT UnifiedEngine.software_reset(), which follows the
        # write with wait_queue() -- guaranteed to time out and print an error on an
        # engine that is still draining -- and then init_unified_engine(), whose
        # 16 KB DRAM self-test at DRAM_START_ADDR would land on live model memory
        # this late in the run. The register write alone has no DRAM side effects.
        SW_RESET_CMD = 0x80008000
        for i, ue in enumerate(self.engines):
            if not ue.is_queue_busy():
                continue
            print(f"    [engines] engine {i} is stuck busy from a previous run; "
                  f"issuing SW_RESET")
            ue.write_reg32(user_dma_core.UE_QUEUE_CTRL_ADDR, SW_RESET_CMD)
            for _ in range(50):                      # ~0.5 s, 10 ms granularity
                if not ue.is_queue_busy():
                    break
                time.sleep(0.01)
            assert not ue.is_queue_busy(), (
                f"engine {i} still reports queue_busy after SW_RESET. It cannot run "
                f"this stage, and continuing would let the primary pass every "
                f"rendezvous on stale flags -- producing fast, WRONG results rather "
                f"than a hang. Power-cycle or reload the bitstream.")

        for ue in self.engines:
            assert not ue.is_capture_on, "preclear_flags() must run outside capture"
            ue.start_capture()
            ue.generate_instruction_flag_clear()
            ue.generate_instruction_halt()
            ue.stop_capture()
            addr = ue.get_program_dram_addr()
            ue.write_captured_instructions_to_dram(addr)
            ue.allocate_program_dram(ue.get_capture_instruction_size_bytes())
            ue.clear_capture_buffer()
            ue.start_execute_from_dram(addr)
            ue.wait_queue(timeout_seconds)

    def sharded_region(self, M: int, body: Callable[[ShardContext], None],
                       join: bool = True) -> None:
        """Replay ``body(ctx)`` once per engine inside one barrier-free region.

        ``body`` must be pure emission: it may read Python state but must emit
        only through ``ctx.ue`` and address memory only through ``ctx``.
        """
        contexts = self.begin_sharded(M)
        for ctx in contexts:
            body(ctx)
        self.end_sharded(join=join)

    # ======================================================================
    # NEW: master/worker sharding -- private arenas, materialized weight
    # shards, persistent worker programs. All of it is additive: nothing here
    # is reachable from the region API (sharded_region / col_sharded_region /
    # k_sharded_region / head_sharded_region), which is unchanged.
    # ======================================================================
    def _require_private_map(self, what: str) -> None:
        assert self.arena is not None, (
            f"{what} needs the private low-DRAM map: construct the scheduler with "
            f"worker_map='private_low'. Under the legacy map there is no per-engine "
            f"arena to place shards in that is guaranteed not to alias the model.")

    def _alloc_private(self, engine_idx: int, size_bytes: int, what: str) -> int:
        """Bump-allocate in an engine's private WEIGHT arena (delegates to the arena)."""
        self._require_private_map(what)
        return self.arena.alloc_weights(engine_idx, size_bytes, what)

    def _alloc_private_tensor(self, engine_idx: int, size_bytes: int, what: str) -> int:
        """Bump-allocate in an engine's private TENSOR window (delegates to the arena)."""
        self._require_private_map(what)
        return self.arena.alloc_tensor(engine_idx, size_bytes, what)

    def private_usage(self) -> list[int]:
        """Bytes of private weight arena used per engine."""
        self._require_private_map("private_usage")
        return self.arena.usage()

    def shard_quantized_weight(self, name: str, main_weight_addr: int, main_scale_addr: int,
                               K: int, N: int, layers: int, main_layer_stride: int,
                               data_type=None, remainder: str = "trailing",
                               max_engines: Optional[int] = None,
                               verbose: bool = True) -> ShardedWeight:
        """MATERIALIZE each engine's column block of a weight into its private arena.

        The full weight already lives in the primary's params region as a quantized
        ``[N, K]`` blob (row-major in N) plus a ``[N, K/64]`` bf16 scale blob, with
        ``main_layer_stride`` bytes between layers. A column shard is therefore a
        CONTIGUOUS ROW BLOCK of both -- which is exactly why N is the axis worth
        splitting: the copy is a byte-range move needing no repacking.

        See the module-level comment on materialized vs zero-copy shards for when to
        use this instead of ``ColumnShardContext.b_addr``. In one line: use this when a
        HARDWARE-LOOPED body must step layer to layer, because only a private packed
        arena makes that stride a single add_imm on a runtime cursor.

        The copy goes card -> host -> card: the source is the image the model loader
        already wrote to DRAM, and there is no device-to-device DMA path.

        Pass ``data_type=DENSE_BF16`` (and ``main_scale_addr=None``) for an
        unquantized bf16 weight: the row-block slicing is identical, there is
        simply no scale blob to carry -- see :meth:`shard_bf16_weight`.
        """
        self._require_private_map("shard_quantized_weight")
        if data_type is None:
            data_type = user_dma_core.TYPE.IF4
        if name in self._weights:
            raise ValueError(f"weight {name!r} already sharded")
        eb = _weight_elem_bytes(data_type)
        has_scale = data_type is not DENSE_BF16
        if has_scale and main_scale_addr is None:
            raise ValueError(f"{name}: {data_type!r} needs main_scale_addr")
        if (K * eb) % 1:
            raise ValueError(f"{name}: K={K} x {eb} B/elem is not a whole number of bytes")
        assert not has_scale or K % COL_ALIGN == 0, (
            f"{name}: K={K} must be a multiple of {COL_ALIGN} -- the scale blob is "
            f"blocked at whole K-vectors, so a column shard's scale stride is only "
            f"linear when K is too")

        splits = self.split_cols(N, remainder=remainder, max_engines=max_engines)
        sw = ShardedWeight(name=name, K=K, N=N, layers=layers, data_type=data_type)

        # PRE-FLIGHT: check every engine has room BEFORE copying a single byte. Without
        # this the first engines get written and the failure lands mid-scatter, leaving
        # some arenas holding this weight and others not -- a state that reads as
        # garbage rather than as an error.
        for engine_idx, (col_offset, cols) in enumerate(splits):
            need = int(cols * K * eb) * layers
            if has_scale:
                need += (cols * K // COL_ALIGN) * 2 * layers
            free = (self.regions[engine_idx].weight_limit
                    - self.arena._weight_cursor[engine_idx])
            if need > free:
                raise MemoryError(
                    f"{name}: engine {engine_idx} needs {need / 2**20:.1f} MB for its "
                    f"{cols}-column shard but only {free / 2**20:.1f} MB is left in its "
                    f"{self.arena.weight_bytes() / 2**20:.0f} MB weight "
                    f"arena (num_engines={self.num_engines}). Already allocated: "
                    f"{', '.join(sorted(self._weights)) or 'nothing'}. Either shard "
                    f"fewer ops or use fewer cores (bigger window each).")

        for engine_idx, (col_offset, cols) in enumerate(splits):
            w_stride = int(cols * K * eb)                      # this shard, one layer
            s_stride = (cols * K // COL_ALIGN) * 2 if has_scale else 0
            w_addr = self._alloc_private(engine_idx, w_stride * layers, f"{name} weights")
            s_addr = (self._alloc_private(engine_idx, s_stride * layers, f"{name} scales")
                      if has_scale else 0)
            sw.shards.append(WeightShard(
                engine_idx=engine_idx, col_offset=col_offset, cols=cols,
                weight_addr=w_addr, scale_addr=s_addr,
                layer_stride=w_stride, scale_layer_stride=s_stride))

        # One layer at a time: read the full row block for this layer, scatter the slices.
        for layer in range(layers):
            src_w = main_weight_addr + layer * main_layer_stride
            src_s = (main_scale_addr + layer * main_layer_stride) if has_scale else 0
            for shard in sw.shards:
                w_off = int(shard.col_offset * K * eb)
                self._copy_dram_bytes(src_w + w_off,
                                      shard.weight_addr + layer * shard.layer_stride,
                                      shard.layer_stride)
                if not has_scale:
                    continue
                s_off = (shard.col_offset * K // COL_ALIGN) * 2
                self._copy_dram_bytes(src_s + s_off,
                                      shard.scale_addr + layer * shard.scale_layer_stride,
                                      shard.scale_layer_stride)

        self._weights[name] = sw
        if verbose:
            used = [f"{u / 2**20:.1f}MB" for u in self.private_usage()]
            print(f"  sharded {sw.summary()}; private arenas used: {', '.join(used)}")
        return sw

    def shard_bf16_weight(self, name: str, main_weight_addr: int,
                          K: int, N: int, layers: int, main_layer_stride: int,
                          remainder: str = "trailing",
                          max_engines: Optional[int] = None,
                          verbose: bool = True) -> ShardedWeight:
        """MATERIALIZE column blocks of an UNQUANTIZED bf16 ``[N, K]`` weight.

        For the weights a model deliberately keeps in bf16 (Qwen2.5-VL's v_proj
        and o_proj, where attention accuracy pays for the width). Column shards
        do not care about the element format: the blob is row-major in N, so a
        column block is a contiguous row block either way. The result feeds
        ``matmat_mul_core`` -- ``scale_addr`` is 0 and there is no scale blob.
        """
        return self.shard_quantized_weight(
            name=name, main_weight_addr=main_weight_addr, main_scale_addr=None,
            K=K, N=N, layers=layers, main_layer_stride=main_layer_stride,
            data_type=DENSE_BF16, remainder=remainder, max_engines=max_engines,
            verbose=verbose)

    def _copy_dram_bytes(self, src_addr: int, dst_addr: int, size_bytes: int) -> None:
        """Move a raw byte range within device DRAM, staging through the host.

        Both endpoints are plain DRAM addresses, so this uses the PRIMARY's DMA
        channels regardless of which engine's arena the destination belongs to --
        an engine's register block selects the compute engine, not the memory path.
        """
        buf = bytearray(size_bytes)
        got = self.primary.dma_read(user_dma_core.DMA_DEVICE_C2H, src_addr, buf, size_bytes)
        if got != size_bytes:
            raise IOError(f"shard copy: read 0x{src_addr:X} returned {got} of {size_bytes} B")
        put = self.primary.dma_write(user_dma_core.DMA_DEVICE_H2C, dst_addr, buf, size_bytes)
        if put != size_bytes:
            raise IOError(f"shard copy: write 0x{dst_addr:X} returned {put} of {size_bytes} B")

    def worker_flops(self, sw: ShardedWeight, M: int = 1) -> int:
        """FLOPs contributed by engines 1..N-1 for one invocation of ``sw``.

        The primary's own share comes back from :meth:`emit_primary_matmat`; a caller
        totalling a program's FLOPs adds the two.
        """
        return sum(2 * M * sw.K * s.cols for s in sw.shards[1:])

    # -- primary-side emission ---------------------------------------------
    def emit_primary_prologue(self, sw: ShardedWeight, gpr_w: int, gpr_s: int) -> None:
        """PRIMARY: seed its private weight/scale cursors, BEFORE the layer loop.

        The primary cannot reuse a folded body's existing per-layer offset register:
        that one advances by the SOURCE image's per-layer size, whereas engine 0's
        shard lives in a private arena where consecutive layers are
        ``shard.layer_stride`` apart -- only this engine's columns are stored there,
        packed layer after layer. Hence its own pair of cursors, advanced once per
        iteration by :meth:`emit_primary_layer_advance`.
        """
        shard = sw.shard(0)
        self.primary.generate_instruction_add_set(
            gpr_w, user_dma_core.ue_35bit_addr_shifter(shard.weight_addr))
        self.primary.generate_instruction_add_set(
            gpr_s, user_dma_core.ue_35bit_addr_shifter(shard.scale_addr))

    def emit_primary_layer_advance(self, sw: ShardedWeight, gpr_w: int, gpr_s: int) -> None:
        """PRIMARY: step its private cursors to the next layer's block."""
        shard = sw.shard(0)
        self.primary.generate_instruction_add_imm(
            src_reg_idx=gpr_w,
            immediate_value=user_dma_core.ue_35bit_addr_shifter(shard.layer_stride),
            dst_reg_idx=gpr_w)
        self.primary.generate_instruction_add_imm(
            src_reg_idx=gpr_s,
            immediate_value=user_dma_core.ue_35bit_addr_shifter(shard.scale_layer_stride),
            dst_reg_idx=gpr_s)

    def emit_primary_matmat(self, sw: ShardedWeight, a_addr: int, out_addr: int,
                            gpr_w: int, gpr_s: int,
                            gpr_a: Optional[int] = None, gpr_out: Optional[int] = None,
                            gpr_M_reg: Optional[int] = None, gelu: bool = False,
                            M: int = 1) -> int:
        """Emit ENGINE 0's column block of a sharded matmul into the primary's capture.

        Only engine 0's own columns are computed here; the workers' blocks come from
        :meth:`emit_worker_program`, and the CALLER owns the rendezvous
        (:meth:`release` before, :meth:`join` after) -- deliberately, because a round
        may hold several ops and only the caller knows where its boundaries are.

        Returns engine 0's own FLOPs; add :meth:`worker_flops` for the whole op.
        """
        shard = sw.shard(0)
        ue = self.primary
        out_slice = _shifted(out_addr, shard.col_offset * 2, f"{sw.name} out")
        kwargs = dict(
            M=M, K=sw.K, N=shard.cols,
            A_DRAM_ADDR=a_addr,
            B_DRAM_ADDR=shard.weight_addr,
            OUTPUT_DRAM_ADDR=out_slice,
            SCALE_DRAM_ADDR=shard.scale_addr,
            data_type=sw.data_type, gelu_enable=gelu,
            gpr_b_addr=gpr_w, gpr_scale_addr=gpr_s,
        )
        if gpr_M_reg is not None:
            kwargs["gpr_M_reg"] = gpr_M_reg
        if gpr_a is not None:
            kwargs["gpr_a_addr"] = _const_addr(ue, a_addr, gpr_a)
        if gpr_out is not None:
            kwargs["gpr_out_addr"] = _const_addr(ue, out_slice, gpr_out)
        return ue.quantized_matmat_core(**kwargs) or 0

    def emit_static_matmat(self, ue, engine_idx: int, sw: ShardedWeight, a_addr: int,
                           out_addr: int, gelu: bool = False, M: int = 1,
                           write_back_disable: bool = False) -> int:
        """Emit one engine's column block with LITERAL addresses -- no runtime cursors.

        For a weight with a single "layer" (an LM head) there is nothing to advance
        between rounds, so this takes the static path exactly as an unsharded op would.
        """
        shard = sw.shard(engine_idx)
        return ue.quantized_matmat_core(
            M=M, K=sw.K, N=shard.cols,
            A_DRAM_ADDR=a_addr,
            B_DRAM_ADDR=shard.weight_addr,
            OUTPUT_DRAM_ADDR=_shifted(out_addr, shard.col_offset * 2, f"{sw.name} out"),
            SCALE_DRAM_ADDR=shard.scale_addr,
            data_type=sw.data_type, gelu_enable=gelu,
            write_back_disable=write_back_disable,
        ) or 0

    # -- batch-split decode attention --------------------------------------
    def shard_attention(self, name: str, batch: int, head_dim: int, aligned_seq_len: int,
                        scratch_bytes: int, verbose: bool = True) -> ShardedAttention:
        """Split a unified_attention_core over its BATCH (query) rows, scratch each.

        K and V are NOT sliced: decode GQA reads one shared KV head per group, so every
        query row in the group reads the same cache. Q, the bias and the output are
        per-row and slice with the batch; the SCRATCH is per-engine and comes out of
        private tensor space, because the core stages V-transpose / scores / scaled-Q
        through it and two engines sharing one would corrupt each other.

        Rows fill from engine 0 UPWARDS -- the opposite of the column splits, and for a
        different question. A column split asks who carries the remainder, and the
        answer is "not the master". This asks which engines PARTICIPATE AT ALL: with
        batch=4 on 8 engines, filling from the front means "4 doing attention, 4 spare"
        rather than a hole at engine 0. A zero-row engine still runs the round's
        handshake -- skipping a rendezvous would desynchronise the group permanently --
        it simply emits no attention core.
        """
        self._require_private_map("shard_attention")
        sa = ShardedAttention(name=name, batch=batch, head_dim=head_dim,
                              aligned_seq_len=aligned_seq_len)
        base, rem = divmod(batch, self.num_engines)
        counts = [base + (1 if i < rem else 0) for i in range(self.num_engines)]
        offsets = [sum(counts[:i]) for i in range(self.num_engines)]
        for engine_idx, (b_off, b_cnt) in enumerate(zip(offsets, counts)):
            addr = self._alloc_private_tensor(engine_idx, scratch_bytes, f"{name} scratch")
            sa.shards.append(AttentionShard(engine_idx=engine_idx, batch_offset=b_off,
                                            batch_rows=b_cnt, scratch_addr=addr))
        if verbose:
            print(f"  sharded {sa.summary()}; scratch {scratch_bytes / 2**20:.2f} MB/engine "
                  f"in private tensor space")
        return sa

    def emit_attention(self, ue, engine_idx: int, op: AttentionOp,
                       gpr_batch: int, gpr_aligned: int, gpr_scale: int,
                       gpr_k: int, gpr_v: int, gpr_q: int, gpr_bias: int,
                       gpr_out: int, gpr_tmp: int) -> int:
        """Emit ONE engine's batch slice of a unified_attention_core.

        Slicing, per dimension:
          Q, OUT  ``[batch, head_dim]``    -> + batch_offset * head_dim * 2, a literal
          BIAS    ``[batch, aligned_seq]`` -> + batch_offset * aligned * 2, and
                                              ``aligned`` is only known at RUN time, so
                                              this offset is computed ON DEVICE
                                              (reg_mul_imm, then add_imm)
          K, V                             -> NOT sliced (shared KV head)
          SCRATCH                          -> this engine's private buffer

        An engine with zero rows emits nothing; it still runs the round's handshake.
        """
        shard = op.sa.shard(engine_idx)
        if shard.batch_rows == 0:
            return 0
        H, bpe = op.sa.head_dim, 2
        row_bytes = H * bpe
        ue.generate_instruction_add_set(gpr_batch, shard.batch_rows)
        _const_addr(ue, op.q_addr + shard.batch_offset * row_bytes, gpr_q)
        _const_addr(ue, op.out_addr + shard.batch_offset * row_bytes, gpr_out)
        if shard.batch_offset:
            # bias row stride is the RUNTIME aligned KV length:
            #   gpr_bias = base + batch_offset * aligned * 2
            ue.generate_instruction_reg_mul_imm(
                gpr_tmp, gpr_aligned,
                user_dma_core.ue_35bit_addr_shifter(shard.batch_offset * bpe))
            ue.generate_instruction_add_imm(
                src_reg_idx=gpr_tmp,
                immediate_value=user_dma_core.ue_35bit_addr_shifter(op.bias_addr),
                dst_reg_idx=gpr_bias)
        else:
            _const_addr(ue, op.bias_addr, gpr_bias)
        return ue.unified_attention_core(
            batch=shard.batch_rows,
            aligned_seq_len=op.sa.aligned_seq_len,
            head_dim=H,
            Q_DRAM_ADDR=op.q_addr + shard.batch_offset * row_bytes,
            K_DRAM_ADDR=op.k_addr, V_DRAM_ADDR=op.v_addr,
            BIAS_DRAM_ADDR=op.bias_addr,
            OUTPUT_DRAM_ADDR=op.out_addr + shard.batch_offset * row_bytes,
            SCRATCH_DRAM_ADDR=shard.scratch_addr,
            IDENTITY_DRAM_ADDR=op.identity_addr,
            gpr_batch_reg=gpr_batch, gpr_aligned_seq_len_reg=gpr_aligned,
            gpr_q_addr=gpr_q, gpr_k_addr=gpr_k, gpr_v_addr=gpr_v,
            gpr_bias_addr=gpr_bias, gpr_out_addr=gpr_out,
            gpr_scale_reg=gpr_scale,
        ) or 0

    def attention_worker_flops(self, sa: ShardedAttention) -> int:
        """FLOPs the WORKERS contribute to one attention invocation (engines 1..N-1)."""
        per_row = 2 * 2 * sa.aligned_seq_len * sa.head_dim   # Q@K^T + P@V
        return sum(s.batch_rows * per_row for s in sa.shards[1:])

    # -- persistent worker programs ----------------------------------------
    def emit_worker_program(self, rounds, layers: int, tail_rounds=()) -> None:
        """Compile and upload each worker's WHOLE program: a hardware-looped body.

        This is the asymmetric counterpart to the region API. Instead of replaying the
        primary's body once per engine, each worker gets ONE captured body that the
        hardware loops ``layers`` times, meeting the master at a :meth:`release` /
        :meth:`join` pair once per round.

        ``rounds`` is a list of ROUNDS; each round is a list of ops, and each round is
        ONE rendezvous. An op is either ``(ShardedWeight, a_addr, out_addr, gelu)`` or
        an :class:`AttentionOp`. Ops inside a round run back-to-back with NO barrier
        between them -- legal only when they all stay in this engine's lane, i.e. none
        reads a column another engine produced. A gate/up pair qualifies (both read the
        same normalized input and write their own output slice); a down-projection does
        not, because its K spans the whole gate*up product, so it needs its own round
        after the master has done the multiply.

        ``tail_rounds`` has the same shape and runs ONCE after the layer loop -- an LM
        head, say: once per token, layer-independent, so it uses literal addresses and
        needs no cursor.

        At the end of each iteration every op's private weight/scale cursor advances by
        that op's OWN packed stride -- never by the source image's per-layer size.
        """
        self._require_private_map("emit_worker_program")
        self._latch_rendezvous("master_worker")
        for idx in self.worker_indices():
            ue = self.engines[idx]
            ue.clear_inst_id()
            # A worker's program is re-emitted WHOLE on every compile, and a model may
            # compile more than once per process (e.g. a profiling pass). alloc_isa_reg's
            # counter is cumulative, so without this reset the second pass starts high
            # and the trip-count registers fall out of the narrow 1..15 window the PBI
            # row loop requires. The master releases its registers instead; a worker owns
            # its whole stream, so resetting is both simpler and idempotent.
            ue.reset_isa_reg_counter()
            ue.start_capture()
            ue.generate_instruction_flag_clear()

            # THE ROW-LOOP TRIP COUNT MUST BE ALLOCATED FIRST. gpr_M_reg lands in a
            # NARROW instruction field, so it has to sit in the low register window
            # (1..15); a worker's counter starts at 1, so allocating it first puts it at
            # r1. Passing a dimension GPR is also what selects quantized_matmat_core's
            # DYNAMIC path -- gpr_*_addr alone is rejected, and the static path could not
            # take a runtime weight base anyway.
            gpr_one = ue.alloc_isa_reg()
            gpr_cnt = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(gpr_one, 1)
            ue.generate_instruction_add_set(gpr_cnt, layers)

            # Attention needs the RUNTIME aligned KV length, which changes every token.
            # A worker program is compiled once, so the value arrives the same way the
            # master gets it: a tiny per-token preamble writes add_set into this register
            # and jumps into the body (see start_workers).
            _attn = [op for rnd in list(rounds) + list(tail_rounds) for op in rnd
                     if isinstance(op, AttentionOp)]
            gpr_aligned = gpr_batch = gpr_scale = None
            gpr_k = gpr_v = gpr_q = gpr_bias = gpr_out = gpr_tmp = gpr_kv = None
            if _attn:
                gpr_aligned = ue.alloc_isa_reg()
                gpr_batch = ue.alloc_isa_reg()
                gpr_scale = ue.alloc_isa_reg()
                gpr_kv = ue.alloc_isa_reg()
                gpr_k, gpr_v = ue.alloc_isa_reg(), ue.alloc_isa_reg()
                gpr_q, gpr_bias = ue.alloc_isa_reg(), ue.alloc_isa_reg()
                gpr_out, gpr_tmp = ue.alloc_isa_reg(), ue.alloc_isa_reg()
                ue.generate_instruction_add_set(gpr_scale, _attn[0].scale_bf16)
                ue.generate_instruction_add_set(gpr_kv, 0)   # per-layer KV cache offset
            self._persistent_aligned_reg[idx] = gpr_aligned

            # One private weight/scale cursor pair per op, across every round: each arena
            # packs only THIS engine's columns, so every op steps by its own shard stride.
            cursors = {}
            for rnd in rounds:
                for op in rnd:
                    if isinstance(op, AttentionOp):
                        continue
                    sw = op[0]
                    if idx >= len(sw.shards):
                        raise NotImplementedError(
                            f"{sw.name} was sharded over {len(sw.shards)} of "
                            f"{self.num_engines} engine(s) (max_engines), and the "
                            f"folded worker-program emitter has no way to skip an "
                            f"op for one engine inside a shared round body. Emit "
                            f"the worker streams directly (begin_worker_round / "
                            f"end_worker_round) for partially-sharded weights.")
                    shard = sw.shard(idx)
                    gpr_w = ue.alloc_isa_reg()
                    gpr_s = ue.alloc_isa_reg()
                    ue.generate_instruction_add_set(
                        gpr_w, user_dma_core.ue_35bit_addr_shifter(shard.weight_addr))
                    ue.generate_instruction_add_set(
                        gpr_s, user_dma_core.ue_35bit_addr_shifter(shard.scale_addr))
                    cursors[id(sw)] = (shard, gpr_w, gpr_s)

            ue.pad_capture_to_64b_boundary()
            body_word_addr = user_dma_core.ue_35bit_addr_shifter(
                ue.get_program_dram_addr()
                + ue.capture_count * user_dma_core.INSTRUCTION_SIZE_BYTES)

            # --- per-layer body: one four-phase rendezvous per round ---
            for rnd in rounds:
                ue.generate_instruction_flag_check_set(target_engine_idx=0)    # 1: go
                for op in rnd:
                    if isinstance(op, AttentionOp):
                        # K/V live in this layer's slice of the shared KV cache: base +
                        # gpr_kv, the worker's own copy of the master's KV layer offset.
                        _offset_addr(ue, gpr_kv, op.k_addr, gpr_k)
                        _offset_addr(ue, gpr_kv, op.v_addr, gpr_v)
                        self.emit_attention(ue, idx, op, gpr_batch, gpr_aligned, gpr_scale,
                                            gpr_k, gpr_v, gpr_q, gpr_bias, gpr_out, gpr_tmp)
                        continue
                    sw, a_addr, out_addr, gelu = op
                    shard, gpr_w, gpr_s = cursors[id(sw)]
                    ue.quantized_matmat_core(
                        M=1, K=sw.K, N=shard.cols,
                        A_DRAM_ADDR=a_addr,
                        B_DRAM_ADDR=shard.weight_addr,
                        OUTPUT_DRAM_ADDR=_shifted(out_addr, shard.col_offset * 2,
                                                  f"{sw.name} out"),
                        SCALE_DRAM_ADDR=shard.scale_addr,
                        data_type=sw.data_type, gelu_enable=gelu,
                        gpr_M_reg=gpr_one, gpr_b_addr=gpr_w, gpr_scale_addr=gpr_s,
                    )
                ue.generate_instruction_flag_set()                             # 2: done
                ue.generate_instruction_flag_check_clear(target_engine_idx=0)  # 3: closed
                ue.generate_instruction_flag_clear()                           # 4: re-armed

            if _attn:
                ue.generate_instruction_add_imm(
                    src_reg_idx=gpr_kv,
                    immediate_value=user_dma_core.ue_35bit_addr_shifter(
                        _attn[0].kv_layer_stride),
                    dst_reg_idx=gpr_kv)
            for shard, gpr_w, gpr_s in cursors.values():
                ue.generate_instruction_add_imm(
                    src_reg_idx=gpr_w,
                    immediate_value=user_dma_core.ue_35bit_addr_shifter(shard.layer_stride),
                    dst_reg_idx=gpr_w)
                ue.generate_instruction_add_imm(
                    src_reg_idx=gpr_s,
                    immediate_value=user_dma_core.ue_35bit_addr_shifter(
                        shard.scale_layer_stride),
                    dst_reg_idx=gpr_s)
            ue.generate_instruction_add_dec(reg_idx=gpr_cnt)
            ue.generate_instruction_jump_abs_jnz(body_word_addr, gpr_cnt)

            # --- once-per-token tail rounds (LM head): literal addresses, no cursors ---
            for rnd in tail_rounds:
                ue.generate_instruction_flag_check_set(target_engine_idx=0)    # 1
                for sw, a_addr, out_addr, gelu in rnd:
                    # Writeback ENABLED: global_argmax() has to read these values back,
                    # because the per-engine argmax registers give indices with no values.
                    self.emit_static_matmat(ue, idx, sw, a_addr, out_addr, gelu=gelu,
                                            write_back_disable=False)
                ue.generate_instruction_flag_set()                             # 2
                ue.generate_instruction_flag_check_clear(target_engine_idx=0)  # 3
                ue.generate_instruction_flag_clear()                           # 4

            ue.generate_instruction_halt()
            ue.stop_capture()

            addr = ue.get_program_dram_addr()
            size = ue.get_capture_instruction_size_bytes()
            self._check_isa_fits(idx, addr, size)
            ue.write_captured_instructions_to_dram(addr)
            ue.allocate_program_dram(size)
            ue.clear_capture_buffer()
            self._persistent_prog[idx] = addr
            self._persistent_body_word[idx] = user_dma_core.ue_35bit_addr_shifter(addr)
            # Slot for the per-token preamble, rewritten by start_workers when a runtime
            # value is passed. Reserved even when it is not, so ISA accounting is uniform.
            self._persistent_preamble[idx] = ue.get_program_dram_addr()
            ue.allocate_program_dram(4 * user_dma_core.INSTRUCTION_SIZE_BYTES)

    def reset_workers(self) -> None:
        """Clear stale flags left set by an aborted run, so the next rendezvous is clean.

        A worker killed mid-rendezvous leaves its done-flag raised; the next run's master
        would then sail through its first CHECK_SET and read a half-written output slice.
        Run this BEFORE launching, not after failing.

        This only clears flags. If an engine is stuck BUSY from a previous process it
        cannot accept this program either -- use :meth:`preclear_flags`, which recovers
        that case with a SW_RESET first.
        """
        for idx in self.worker_indices():
            ue = self.engines[idx]
            ue.clear_inst_id()
            ue.start_capture()
            ue.generate_instruction_flag_clear()
            ue.generate_instruction_halt()
            ue.stop_capture()
            addr = ue.get_program_dram_addr()
            ue.write_captured_instructions_to_dram(addr)
            ue.clear_capture_buffer()
            ue.program_execute(addr, timeout=1.0)

    # -- cross-engine argmax ------------------------------------------------
    def global_argmax(self, sw: ShardedWeight, out_addr: int) -> int:
        """Combine the per-engine argmaxes of a column-sharded output into the global one.

        Each engine ran the matmul over ITS OWN column block, so its argmax register
        holds a LOCAL index into that block, and the hardware exposes only indices --
        there is no max-VALUE register to compare across engines. The global winner is
        therefore found by taking each engine's rank-1 candidate (the maximum of its own
        slice, so the global maximum is guaranteed to be among the N) and reading just
        those values back from DRAM to compare. That is N tiny reads, not a full-width
        readback -- but it does require the sharded op to run with writeback ENABLED,
        unlike a single-engine path that can keep the vector on-chip and read the
        register directly.
        """
        if len(sw.shards) != self.num_engines:
            raise NotImplementedError(
                f"sharded_argmax needs every engine to hold a slice, but {sw.name} "
                f"covers {len(sw.shards)} of {self.num_engines} engine(s); the "
                f"engines without one never produced a candidate")
        best_idx, best_val = None, None
        for i in range(self.num_engines):
            shard = sw.shard(i)
            local = self.engines[i].get_arg_max_index()
            if not 0 <= local < shard.cols:
                raise RuntimeError(
                    f"engine {i} argmax index {local} outside its shard of {shard.cols} "
                    f"columns -- the per-engine argmax register did not track this shard")
            gidx = shard.col_offset + local
            val = self._read_bf16(out_addr + gidx * 2)
            if best_val is None or val > best_val:
                best_idx, best_val = gidx, val
        return best_idx

    def _read_bf16(self, addr: int) -> float:
        """Read one bf16 from DRAM through a 64-byte aligned window (DMA alignment)."""
        base = addr & ~0x3F
        off = addr - base
        buf = bytearray(64)
        got = self.primary.dma_read(user_dma_core.DMA_DEVICE_C2H, base, buf, 64)
        if got != 64:
            raise IOError(f"bf16 read at 0x{base:X} returned {got} of 64 bytes")
        # bf16 is the high 16 bits of an fp32, so widening is exact: append 16 zero bits.
        bits = int.from_bytes(buf[off:off + 2], "little") << 16
        return struct.unpack("<f", bits.to_bytes(4, "little"))[0]

    # -- private-space protection ------------------------------------------
    def _check_isa_fits(self, engine_idx: int, addr: int, size_bytes: int) -> None:
        """Refuse a worker program that would leave its private ISA slice."""
        self.arena.check_isa_fits(engine_idx, addr, size_bytes)

    def verify_private_space(self, verbose: bool = True) -> None:
        """Check every engine's weight arena and ISA slice are inside their window."""
        self._require_private_map("verify_private_space")
        self.arena.verify(self.engines, verbose=verbose)

    # ------------------------------------------------------------ internal --
    def _acquire_m_reg(self, engine_idx: int, rows: int) -> int:
        key = (engine_idx, rows)
        if key not in self._m_regs:
            ue = self.engines[engine_idx]
            reg = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(reg, rows)
            self._m_regs[key] = reg
        return self._m_regs[key]

    def _acquire_n_reg(self, engine_idx: int, cols: int) -> int:
        key = (engine_idx, cols)
        if key not in self._n_regs:
            ue = self.engines[engine_idx]
            reg = ue.alloc_isa_reg()
            ue.generate_instruction_add_set(reg, cols)
            self._n_regs[key] = reg
        return self._n_regs[key]
