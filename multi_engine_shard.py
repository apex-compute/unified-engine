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
             writes into -- see models/pi05_libero/pi05_libero_test.py:2265).
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
"""

from __future__ import annotations

import hashlib
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
                 allow_more_than_two_engines: bool = False):
        assert num_engines >= 1, f"num_engines must be >= 1, got {num_engines}"
        if num_engines > 2:
            # The device caps at 2 engines; N>2 is code-complete but unverified.
            assert allow_more_than_two_engines, (
                f"num_engines={num_engines} > 2 is unverified on this device; "
                f"pass allow_more_than_two_engines=True to opt in"
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

        # Worker engines. Their allocator bases are OWNED HERE: DRAM is one flat
        # space, so independent Python allocators would silently collide with
        # the model's params/tensor/program cursors.
        if worker_dram_base is None:
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
        guard = self._save_dram_selftest_region() if num_engines > 1 else None
        for i in range(1, num_engines):
            base = worker_dram_base + (i - 1) * worker_dram_stride
            assert base != getattr(primary_ue, "_params_dram_base", None), \
                "worker DRAM base collides with the primary's params base"
            self.workers.append(UnifiedEngine(
                BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * engine_base_stride,
                params_dram_base=base,
                tensor_dram_base=base + worker_tensor_offset,
                program_dram_base=base + worker_program_offset,
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
    def split_cols(self, N: int) -> list[tuple[int, int]]:
        """Return [(col_offset, col_count)] per engine for a full column count N.

        Column shards are split by whole ``col_align``-element blocks and are
        deliberately allowed to be UNEVEN (the trailing engines get one block
        less) rather than ever unaligned -- the same trade ``split_rows``
        makes in ``"blocks"`` mode, and here it is the ONLY rule: a partial
        64-element vector is not something the matvec pipeline, the B row
        block, or the scale blob can express.

        At bpe=2 a ``col_align``-element block is exactly 128 bytes, so every
        offset this produces is a whole SRAM row (asserted in :func:`_shifted`).
        """
        n = self.num_engines
        a = self.col_align
        if n == 1:
            return [(0, N)]
        assert N % a == 0, (
            f"split_cols: N={N} is not a multiple of col_align={a}. Column shards "
            f"must be multiples of {a} elements; pad N or shard a different axis.")
        blocks = N // a
        assert blocks >= n, (
            f"split_cols: N={N} is only {blocks} block(s) of {a} columns, too few "
            f"for num_engines={n}")
        base, rem = divmod(blocks, n)
        counts = [a * (base + (1 if i < rem else 0)) for i in range(n)]
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
        self.barrier()
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
        self.barrier()
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
        self._worker_prog_addrs = []
        for w in self.workers:
            w.generate_instruction_halt()
            w.stop_capture()
            addr = w.get_program_dram_addr()
            w.write_captured_instructions_to_dram(addr)
            w.allocate_program_dram(w.get_capture_instruction_size_bytes())
            self._worker_prog_addrs.append(addr)
        self._program_open = False
        return list(self._worker_prog_addrs)

    def start_workers(self, prog_addrs: Optional[list[int]] = None) -> None:
        """Launch every worker program. Call BEFORE launching the primary's
        program, on EVERY execution (each run ends with the workers halted).

        ``prog_addrs`` overrides the most recent finalize()'s addresses -- pass
        the stage's saved list when one scheduler carries several stages."""
        addrs = self._worker_prog_addrs if prog_addrs is None else prog_addrs
        assert len(addrs) == len(self.workers), \
            f"start_workers: {len(addrs)} program address(es) for {len(self.workers)} worker(s)"
        for w, addr in zip(self.workers, addrs):
            w.start_execute_from_dram(addr)

    def worker_program_bytes(self) -> int:
        return sum(w.get_capture_instruction_size_bytes() for w in self.workers)

    # ----------------------------------------------------------- regions ---
    def begin_sharded(self, M: int) -> list[ShardContext]:
        """Rendezvous, then open a sharded region, returning one
        :class:`ShardContext` per engine."""
        assert self._program_open, "begin_sharded() without begin_program()"
        assert not self._in_region, "nested sharded regions are not supported"
        split = self.split_rows(M)
        self.barrier()
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
        if join:
            self.barrier()

    def barrier(self) -> None:
        """Symmetric all-engine rendezvous (see module docstring).

        Every engine raises its own flag, spin-waits on every other engine's,
        waits out the CLEAR margin, then lowers its own -- the shape proven to
        re-arm by ``flag_rendezvous_repeat_test``.
        """
        assert not self._in_region, "barrier() inside an open sharded region"
        if self.num_engines == 1:
            return   # exact passthrough: not one extra instruction
        for i, ue in enumerate(self.engines):
            ue.generate_instruction_flag_set()
            for j in range(self.num_engines):
                if j != i:
                    ue.generate_instruction_flag_check(target_engine_idx=j)
            for _ in range(self.barrier_margin_nops):
                ue.generate_instruction_nop()
            ue.generate_instruction_flag_clear()

    def preclear_flags(self, timeout_seconds: float = 5.0) -> None:
        """Run a tiny program on every engine that just clears its flag.

        Call once before the first execution: a flag left set by an earlier
        program would make the first CHECK pass spuriously.
        """
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
