"""N-sharded quantized matmul for the gemma3 decoder, over up to 8 engines.

This is a deliberately NARROW, gemma3-only companion to the general
``multi_engine_shard.py`` in the repo root. It supports exactly one operation --
``quantized_matmat_core`` split along its OUTPUT COLUMNS (N) -- because that is
the only shape the gemma3 decoder needs: every decoder matmul is ``M == 1``
(one token), so there are no rows to split, and an N-shard needs no cross-engine
reduction (each engine writes a disjoint slice of the output vector).

Today it is wired to the MLP **gate** only; ``mlp up`` and the LM head are the
obvious next users and need no new machinery here, just another
:meth:`Gemma3ShardGroup.shard_quantized_weight` call.

DRAM map
--------
The main core's own regions are UNTOUCHED -- PARAMS / TENSOR / PROGRAM still
start at ``0x8000_0000`` exactly as the single-engine build lays them out. This
module only claims the low 2 GB, which the single-engine gemma3 build never
uses, and splits it EVENLY over however many engines are in play::

    stride = align_down(0x8000_0000 / num_engines, 16 MB)
    core i  ->  [i * stride, (i + 1) * stride)
        weights : base                          .. base + stride - 0x0200_0000
        ISA     : base + stride - 0x0200_0000   .. base + stride - 0x0100_0000   16 MB
        tensor  : base + stride - 0x0100_0000   .. base + stride                 16 MB

The window therefore SCALES with the core count, which is the direction that
matters: fewer cores means fewer, larger shards, and a correspondingly larger
arena to hold them. Concretely::

    --multi-core 2  ->  1024 MB/core   core0 0x0000_0000, core1 0x4000_0000
    --multi-core 4  ->   512 MB/core
    --multi-core 8  ->   256 MB/core   (224 MB of weights, as before)

The stride is rounded DOWN to 16 MB, so a core count that does not divide the
2 GB evenly leaves a little slack at the top rather than ever crossing into the
main core's map at ``0x8000_0000``.

Only the WEIGHTS are genuinely private. A worker reads its activation input
from, and writes its output slice into, the MAIN core's tensor region -- the
whole point of an N-shard is that the engines cooperate on one shared output
vector. The private tensor window is therefore reserved rather than used: it
exists so a worker has a well-formed ``tensor_dram_base`` that can never alias
the main core's activations, and so a future op that needs genuine per-engine
scratch has somewhere to put it.

Column splitting
----------------
A column shard must be a whole multiple of ``UE_VECTOR_SIZE`` (64): the B row
block, the scale blob stride and the matvec pipeline are all expressed in whole
64-element vectors, so a partial vector is not representable. gemma3's
``mlp_elements`` is 6912 == 108 x 64, and 108 does not divide by 8, so an
8-engine split CANNOT be even. Shards are therefore split by whole 64-column
blocks with the remainder handed to the leading engines (4 x 896 + 4 x 832 at
8 engines) -- uneven, but never unaligned. This is the same trade
``multi_engine_shard.MultiEngineScheduler.split_cols`` makes.
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import user_dma_core
from user_dma_core import (TYPE, UE_VECTOR_SIZE, URAM_NEAR_FULL_ELEMENTS, UnifiedEngine,
                           ue_35bit_addr_shifter)

# --------------------------------------------------------------------------
# Map constants
# --------------------------------------------------------------------------
MAX_ENGINES = 8
"""Hard ceiling: the private map below carves the low 2 GB into 8 windows."""

PRIVATE_TOTAL = user_dma_core.DRAM_START_ADDR   # the whole low 2 GB, up to the main map
PRIVATE_ALIGN = 0x0100_0000           # 16 MB window granularity
PRIVATE_TENSOR_BYTES = 0x0100_0000    # 16 MB, the ending slice of each window
PRIVATE_ISA_BYTES = 0x0100_0000       # 16 MB, immediately before the tensor slice


def private_stride(num_engines: int) -> int:
    """Size of each engine's private window: the low 2 GB divided by the core count.

    The windows SCALE with the core count rather than being a fixed 256 MB, so fewer
    cores means a bigger arena each -- which is exactly when each shard is larger. At
    2 cores that is 1 GB apiece (core 0 at 0x0, core 1 at 0x4000_0000); at 8 it is the
    256 MB the map started with. Rounded DOWN to 16 MB so the sum never crosses into
    the main core's map at 0x8000_0000, which is why an odd core count simply leaves a
    little slack at the top rather than overlapping anything.
    """
    if not 1 <= num_engines <= MAX_ENGINES:
        raise ValueError(f"num_engines must be in [1, {MAX_ENGINES}], got {num_engines}")
    stride = (PRIVATE_TOTAL // num_engines) & ~(PRIVATE_ALIGN - 1)
    floor = PRIVATE_TENSOR_BYTES + PRIVATE_ISA_BYTES + PRIVATE_ALIGN
    if stride < floor:
        raise ValueError(
            f"num_engines={num_engines} leaves only 0x{stride:X} per window, below the "
            f"0x{floor:X} needed for ISA + tensor + at least one weight block")
    assert num_engines * stride <= PRIVATE_TOTAL, "private windows must not reach 0x80000000"
    return stride


def private_weight_bytes(num_engines: int) -> int:
    """Weight arena inside one private window (the window minus its ISA + tensor tail)."""
    return private_stride(num_engines) - PRIVATE_TENSOR_BYTES - PRIVATE_ISA_BYTES

ENGINE_BASE_STRIDE = 0x0001_0000
"""AXI-Lite stride between engine register blocks (engine i at UE_0 + i*stride)."""

COL_ALIGN = UE_VECTOR_SIZE            # 64 elements
BPE = 2                               # bf16 bytes per element (activations, scales)

# MASKED: superseded by the four-phase CHECK_SET/CHECK_CLEAR handshake (see
# emit_release_workers). These sized the timing margins the old workaround needed while
# the ISA had no wait-for-zero; nothing emits them now.
# SKEW_MARGIN_ELEMENTS = URAM_NEAR_FULL_ELEMENTS  # 262080
SKEW_MARGIN_ELEMENTS = 3000 * UE_VECTOR_SIZE

ACK_MARGIN_ELEMENTS = 32 * UE_VECTOR_SIZE


def _elem_bytes(data_type) -> float:
    """Bytes per weight element for a quantized blob."""
    if data_type == TYPE.IF4:
        return 0.5
    if data_type == TYPE.IF8:
        return 1.0
    raise AssertionError(
        f"multi_engine_shard_gemma3 supports IF4/IF8 weights only, got {data_type!r}")


# --------------------------------------------------------------------------
# Private DRAM windows
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PrivateRegion:
    """One engine's private 256 MB window in the low 2 GB."""

    engine_idx: int
    base: int
    stride: int
    weight_base: int
    weight_limit: int
    isa_base: int
    tensor_base: int

    @property
    def weight_capacity(self) -> int:
        return self.weight_limit - self.weight_base

    def describe(self) -> str:
        mb = 1024 * 1024
        return (f"core {self.engine_idx}: "
                f"[0x{self.base:08X}..0x{self.base + self.stride:08X}) {self.stride // mb} MB"
                f"  |  weights 0x{self.weight_base:08X}..0x{self.weight_limit:08X} "
                f"{self.weight_capacity // mb} MB"
                f"  |  isa 0x{self.isa_base:08X} {PRIVATE_ISA_BYTES // mb} MB"
                f"  |  tensor 0x{self.tensor_base:08X} {PRIVATE_TENSOR_BYTES // mb} MB")


def private_region(engine_idx: int, num_engines: int) -> PrivateRegion:
    """The private window for ``engine_idx`` (0-based) at a given core count."""
    if not 0 <= engine_idx < num_engines:
        raise ValueError(f"engine_idx must be in [0, {num_engines}), got {engine_idx}")
    stride = private_stride(num_engines)
    base = engine_idx * stride
    isa_base = base + stride - PRIVATE_TENSOR_BYTES - PRIVATE_ISA_BYTES
    return PrivateRegion(
        engine_idx=engine_idx,
        base=base,
        stride=stride,
        weight_base=base,
        weight_limit=isa_base,
        isa_base=isa_base,
        tensor_base=isa_base + PRIVATE_ISA_BYTES,
    )


def describe_map(num_engines: int) -> str:
    """Human-readable dump of the private map, for the run log."""
    lines = [f"multi-core private DRAM map ({num_engines} engine(s), "
             f"{private_stride(num_engines) // (1024 * 1024)} MB per core; "
             f"main PARAMS/TENSOR/PROGRAM at 0x{user_dma_core.DRAM_START_ADDR:08X} unchanged):"]
    lines += ["  " + private_region(i, num_engines).describe() for i in range(num_engines)]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Column splitting
# --------------------------------------------------------------------------
def split_n(N: int, num_engines: int) -> list[tuple[int, int]]:
    """Split ``N`` output columns into ``num_engines`` 64-aligned blocks.

    Returns ``[(col_offset, cols)]``, with the remainder going to the TRAILING engines
    so the highest-numbered slaves carry the extra 64-column blocks and engine 0 -- the
    master, which also runs everything that is not sharded (attention, the norms, the
    gate*up multiply, the residuals) -- gets the smallest shard. Since every round ends
    at a barrier where the master waits for the slowest worker, giving the master less
    matmul is what shortens that wait.

    Never returns an unaligned or empty shard -- both are unrepresentable in the B row
    block / scale stride, so they raise instead.
    """
    if num_engines == 1:
        return [(0, N)]
    if not 1 <= num_engines <= MAX_ENGINES:
        raise ValueError(f"num_engines must be in [1, {MAX_ENGINES}], got {num_engines}")
    if N % COL_ALIGN:
        raise ValueError(
            f"split_n: N={N} is not a multiple of {COL_ALIGN}; a column shard must be "
            f"whole {COL_ALIGN}-element vectors")
    blocks = N // COL_ALIGN
    if blocks < num_engines:
        raise ValueError(
            f"split_n: N={N} is only {blocks} block(s) of {COL_ALIGN}, too few for "
            f"num_engines={num_engines}")
    base, rem = divmod(blocks, num_engines)
    counts = [COL_ALIGN * (base + (1 if i >= num_engines - rem else 0))
              for i in range(num_engines)]
    offsets = [sum(counts[:i]) for i in range(num_engines)]
    return list(zip(offsets, counts))


def can_split(N: int, num_engines: int) -> bool:
    """Whether ``N`` columns can be handed to ``num_engines`` at 64-column granularity.

    False when N has fewer than one 64-block per engine -- gemma3's K/V projections are
    N=256, i.e. 4 blocks, so they cap out at 4 engines however many the board has.
    """
    return num_engines == 1 or (N % COL_ALIGN == 0 and N // COL_ALIGN >= num_engines)


def split_batch(B: int, num_engines: int) -> list[tuple[int, int]]:
    """Split ``B`` batch rows over the engines, filling from engine 0 UPWARDS.

    Unlike a column shard there is no 64-element granularity here -- a batch row is a
    whole Q vector -- so the only floor is one row per engine. gemma3 decode attention has
    ``batch = group_size = 4``, so at 8 engines the first four take one row each and the
    four EXTRA engines get none.

    Note this fills from the front, the opposite of :func:`split_n`. The column splits
    hand the remainder to the trailing engines so the master -- which also runs everything
    unsharded -- carries the smallest share of work every engine participates in. Here the
    question is different: which engines participate AT ALL. Filling from the front keeps
    the idle ones contiguous at the top, so "8 cores" means "4 doing attention, 4 spare"
    rather than a hole at engine 0.

    A zero-row engine still runs the round's handshake -- skipping a rendezvous would
    desynchronise the group permanently -- it simply emits no attention core.
    """
    base, rem = divmod(B, num_engines)
    counts = [base + (1 if i < rem else 0) for i in range(num_engines)]
    offsets = [sum(counts[:i]) for i in range(num_engines)]
    return list(zip(offsets, counts))


@dataclass
class AttentionShard:
    """One engine's slice of a batch-split unified_attention_core.

    ``scratch_addr`` is the piece that has no analogue in the matmul shards: the core
    stages V-transpose / scores / scaled-Q through a SCRATCH buffer, and two engines
    sharing one would corrupt each other. Each gets its own out of its PRIVATE TENSOR
    window -- the 16 MB slice that was reserved and unused until now.
    """

    engine_idx: int
    batch_offset: int
    batch_rows: int
    scratch_addr: int


@dataclass
class AttentionOp:
    """An attention op inside a worker round: everything needed to re-emit the core.

    Addresses are the MAIN core's tensor buffers -- Q, bias and OUT are sliced by this
    engine's batch offset, K/V are shared whole (single KV head), and only SCRATCH is
    private.
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
    """A unified_attention_core distributed over its batch (query-head) dimension."""

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


# --------------------------------------------------------------------------
# Sharded weights
# --------------------------------------------------------------------------
@dataclass
class WeightShard:
    """One engine's column block of one sharded weight, in that engine's private DRAM.

    ``weight_addr`` / ``scale_addr`` are layer 0; layer L is at
    ``weight_addr + L * layer_stride`` (and likewise for the scale). The stride is
    THIS shard's own packed size, not the main image's ``LAYER_WEIGHT_SIZE``:
    a private arena holds only this engine's columns, packed layer after layer.
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
    shards: list[WeightShard] = field(default_factory=list)

    def shard(self, engine_idx: int) -> WeightShard:
        return self.shards[engine_idx]

    def summary(self) -> str:
        parts = ", ".join(f"e{s.engine_idx}:{s.cols}" for s in self.shards)
        return (f"{self.name}: N={self.N} over {len(self.shards)} engine(s) [{parts}] "
                f"K={self.K}, {self.layers} layer(s)")


# --------------------------------------------------------------------------
# The shard group
# --------------------------------------------------------------------------
class Gemma3ShardGroup:
    """Engine 0 (the caller's own engine) plus ``num_engines - 1`` worker engines.

    The primary is passed in rather than constructed: it is the live
    ``Gemma3_UnifiedEngine`` that already owns the model, the weights and the main
    DRAM map. Workers are plain :class:`UnifiedEngine` handles pointed at their own
    register block and their own private DRAM window; they hold nothing but weight
    shards and their own instruction stream.
    """

    def __init__(self, primary, num_engines: int, verbose: bool = True):
        if not 1 <= num_engines <= MAX_ENGINES:
            raise ValueError(f"num_engines must be in [1, {MAX_ENGINES}], got {num_engines}")
        hw_cores = user_dma_core.ANDROMEDA_CORE_COUNT
        if hw_cores is not None and num_engines > hw_cores:
            raise ValueError(
                f"--multi-core {num_engines} exceeds the {hw_cores} core(s) HW_INFO reports "
                f"for the loaded bitstream")

        self.num_engines = num_engines
        self.primary = primary
        self.regions = [private_region(i, num_engines) for i in range(num_engines)]
        self.engines: list = [primary]
        for i in range(1, num_engines):
            region = self.regions[i]
            self.engines.append(UnifiedEngine(
                BASE_ADDR=user_dma_core.UE_0_BASE_ADDR + i * ENGINE_BASE_STRIDE,
                params_dram_base=region.weight_base,
                tensor_dram_base=region.tensor_base,
                program_dram_base=region.isa_base,
            ))
        # Next free byte in each engine's private weight arena, and in its private TENSOR
        # window -- the latter is used only by attention scratch, which is the one thing an
        # engine cannot share with its peers.
        self._weight_cursor = [r.weight_base for r in self.regions]
        self._tensor_cursor = [r.tensor_base for r in self.regions]
        self._weights: dict[str, ShardedWeight] = {}
        self._programs: dict[int, int] = {}       # engine_idx -> worker body addr
        self._preamble: dict[int, int] = {}       # engine_idx -> per-token preamble addr
        self._body_word_addr: dict[int, int] = {}
        self._gpr_aligned: dict[int, Optional[int]] = {}
        if verbose:
            print(describe_map(num_engines))

    # -- workers ------------------------------------------------------------
    @property
    def workers(self) -> list:
        """Engines 1..n-1 (the primary is not a worker, it runs the main program)."""
        return self.engines[1:]

    def worker_indices(self) -> list[int]:
        return list(range(1, self.num_engines))

    # -- private weight arena ----------------------------------------------
    def _alloc_private(self, engine_idx: int, size_bytes: int, what: str) -> int:
        """Bump-allocate ``size_bytes`` in an engine's private weight arena, 64 B aligned."""
        addr = (self._weight_cursor[engine_idx] + 63) & ~63
        end = addr + size_bytes
        limit = self.regions[engine_idx].weight_limit
        if end > limit:
            raise MemoryError(
                f"{what}: engine {engine_idx} private weight arena overflow -- needs "
                f"0x{end:X}, window ends at 0x{limit:X} "
                f"({private_weight_bytes(self.num_engines) // (1024 * 1024)} MB per core "
                f"at --multi-core {self.num_engines})")
        self._weight_cursor[engine_idx] = end
        return addr

    def _alloc_private_tensor(self, engine_idx: int, size_bytes: int, what: str) -> int:
        """Bump-allocate in an engine's private TENSOR window, 64 B aligned."""
        addr = (self._tensor_cursor[engine_idx] + 63) & ~63
        limit = self.regions[engine_idx].tensor_base + PRIVATE_TENSOR_BYTES
        if addr + size_bytes > limit:
            raise MemoryError(
                f"{what}: engine {engine_idx} private tensor window overflow -- needs "
                f"0x{addr + size_bytes:X}, window ends at 0x{limit:X} "
                f"({PRIVATE_TENSOR_BYTES // 2**20} MB per core)")
        self._tensor_cursor[engine_idx] = addr + size_bytes
        return addr

    def shard_attention(self, name: str, batch: int, head_dim: int, aligned_seq_len: int,
                        scratch_bytes: int, verbose: bool = True) -> ShardedAttention:
        """Split a unified_attention_core over its batch rows, one scratch buffer each.

        K and V are NOT sliced: gemma3 decode is GQA with a single KV head, so every query
        row in the group reads the same cache. Q, the bias and the output are per-row and
        slice with the batch; the scratch is per-engine and comes out of private tensor
        space.
        """
        sa = ShardedAttention(name=name, batch=batch, head_dim=head_dim,
                              aligned_seq_len=aligned_seq_len)
        for engine_idx, (b_off, b_cnt) in enumerate(split_batch(batch, self.num_engines)):
            addr = self._alloc_private_tensor(engine_idx, scratch_bytes, f"{name} scratch")
            sa.shards.append(AttentionShard(engine_idx=engine_idx, batch_offset=b_off,
                                            batch_rows=b_cnt, scratch_addr=addr))
        if verbose:
            print(f"  sharded {sa.summary()}; scratch {scratch_bytes / 2**20:.2f} MB/engine "
                  f"in private tensor space")
        return sa

    def private_usage(self) -> list[int]:
        """Bytes of private weight arena used per engine."""
        return [self._weight_cursor[i] - self.regions[i].weight_base
                for i in range(self.num_engines)]

    # -- weight sharding ----------------------------------------------------
    def shard_quantized_weight(self, name: str, main_weight_addr: int, main_scale_addr: int,
                               K: int, N: int, layers: int, main_layer_stride: int,
                               data_type=TYPE.IF4, verbose: bool = True) -> ShardedWeight:
        """Copy each engine's column block of a weight into that engine's private DRAM.

        The full weight already lives in the MAIN core's params region as a quantized
        ``[N, K]`` blob (row-major in N) plus a ``[N, K/64]`` bf16 scale blob, with
        ``main_layer_stride`` bytes between layers. A column shard is therefore a
        contiguous row block of both -- which is exactly why N is the axis worth
        splitting: the copy is a byte-range move, needing no repacking.

        The copy goes card -> host -> card, because the source is the main image the
        model loader already wrote to DRAM; there is no device-to-device DMA path.
        """
        if name in self._weights:
            raise ValueError(f"weight {name!r} already sharded")
        eb = _elem_bytes(data_type)
        if (K * eb) % 1:
            raise ValueError(f"{name}: K={K} x {eb} B/elem is not a whole number of bytes")

        splits = split_n(N, self.num_engines)
        sw = ShardedWeight(name=name, K=K, N=N, layers=layers, data_type=data_type)

        # PRE-FLIGHT: check every engine has room BEFORE copying a single byte. Without
        # this the first engines get written and the failure lands mid-scatter, leaving
        # some arenas holding this weight and others not -- a state that reads as garbage
        # rather than as an error.
        for engine_idx, (col_offset, cols) in enumerate(splits):
            need = int(cols * K * eb) * layers + (cols * K // COL_ALIGN) * BPE * layers
            free = self.regions[engine_idx].weight_limit - self._weight_cursor[engine_idx]
            if need > free:
                raise MemoryError(
                    f"{name}: engine {engine_idx} needs {need / 2**20:.1f} MB for its "
                    f"{cols}-column shard but only {free / 2**20:.1f} MB is left in its "
                    f"{private_weight_bytes(self.num_engines) / 2**20:.0f} MB weight arena "
                    f"(--multi-core {self.num_engines}). Already allocated: "
                    f"{', '.join(sorted(self._weights))or 'nothing'}. Either shard fewer "
                    f"ops or use fewer cores (bigger window each).")

        for engine_idx, (col_offset, cols) in enumerate(splits):
            w_stride = int(cols * K * eb)                      # this shard, one layer
            s_stride = (cols * K // COL_ALIGN) * BPE
            w_addr = self._alloc_private(engine_idx, w_stride * layers, f"{name} weights")
            s_addr = self._alloc_private(engine_idx, s_stride * layers, f"{name} scales")
            sw.shards.append(WeightShard(
                engine_idx=engine_idx, col_offset=col_offset, cols=cols,
                weight_addr=w_addr, scale_addr=s_addr,
                layer_stride=w_stride, scale_layer_stride=s_stride))

        # One layer at a time: read the full row block for this layer, scatter the slices.
        for layer in range(layers):
            src_w = main_weight_addr + layer * main_layer_stride
            src_s = main_scale_addr + layer * main_layer_stride
            for shard in sw.shards:
                w_off = int(shard.col_offset * K * eb)
                s_off = (shard.col_offset * K // COL_ALIGN) * BPE
                self._copy_bytes(src_w + w_off, shard.weight_addr + layer * shard.layer_stride,
                                 shard.layer_stride)
                self._copy_bytes(src_s + s_off, shard.scale_addr + layer * shard.scale_layer_stride,
                                 shard.scale_layer_stride)

        self._weights[name] = sw
        if verbose:
            used = [f"{u / (1024 * 1024):.1f}MB" for u in self.private_usage()]
            print(f"  sharded {sw.summary()}; private arenas used: {', '.join(used)}")
        return sw

    def _copy_bytes(self, src_addr: int, dst_addr: int, size_bytes: int) -> None:
        """Move a raw byte range within device DRAM, staging through the host.

        Both endpoints are plain DRAM addresses, so this uses the primary's DMA
        channels regardless of which engine's arena the destination belongs to --
        engine register blocks select the compute engine, not the memory path.
        """
        buf = bytearray(size_bytes)
        got = self.primary.dma_read(user_dma_core.DMA_DEVICE_C2H, src_addr, buf, size_bytes)
        if got != size_bytes:
            raise IOError(f"shard copy: read 0x{src_addr:X} returned {got} of {size_bytes} bytes")
        put = self.primary.dma_write(user_dma_core.DMA_DEVICE_H2C, dst_addr, buf, size_bytes)
        if put != size_bytes:
            raise IOError(f"shard copy: write 0x{dst_addr:X} returned {put} of {size_bytes} bytes")

    # -- the sharded op -----------------------------------------------------
    def emit_static_matmat(self, ue, engine_idx: int, sw: ShardedWeight, a_addr: int,
                           out_addr: int, gelu: bool = False,
                           write_back_disable: bool = False) -> int:
        """Emit one engine's column block with LITERAL addresses -- no runtime cursors.

        For a weight with a single "layer" (the LM head), there is nothing to advance
        between rounds, so this takes the legacy static path exactly as the unsharded
        LM head does.
        """
        shard = sw.shard(engine_idx)
        return ue.quantized_matmat_core(
            M=1, K=sw.K, N=shard.cols,
            A_DRAM_ADDR=a_addr,
            B_DRAM_ADDR=shard.weight_addr,
            OUTPUT_DRAM_ADDR=out_addr + shard.col_offset * BPE,
            SCALE_DRAM_ADDR=shard.scale_addr,
            data_type=sw.data_type, gelu_enable=gelu,
            write_back_disable=write_back_disable,
        ) or 0

    def global_argmax(self, sw: ShardedWeight, logits_addr: int) -> int:
        """Combine the per-engine argmaxes of a sharded LM head into the global one.

        Each engine ran the matmul over ITS OWN column block, so its UE_ARGMAX1_INDEX
        register holds a LOCAL index into that block, and the hardware exposes only
        indices -- there is no max-VALUE register to compare across engines. The global
        winner is therefore found by taking each engine's rank-1 candidate (the maximum of
        its own slice, so the global maximum is guaranteed to be among the 8) and reading
        just those logits back from DRAM to compare. That is 8 tiny reads per token, not
        the 512 KB a full-logits readback would cost -- but it does require the sharded LM
        head to run with writeback ENABLED, unlike the single-engine path which keeps the
        logits on-chip and reads the register directly.
        """
        best_idx, best_val = None, None
        for i in range(self.num_engines):
            shard = sw.shard(i)
            local = self.engines[i].get_arg_max_index()
            if not 0 <= local < shard.cols:
                raise RuntimeError(
                    f"engine {i} argmax index {local} outside its shard of {shard.cols} "
                    f"columns -- the per-engine argmax register did not track this shard")
            gidx = shard.col_offset + local
            val = self._read_bf16(logits_addr + gidx * BPE)
            if best_val is None or val > best_val:
                best_idx, best_val = gidx, val
        return best_idx

    def _read_bf16(self, addr: int) -> float:
        """Read one bf16 from DRAM via a 64-byte aligned window (DMA alignment)."""
        base = addr & ~0x3F
        off = addr - base
        buf = bytearray(64)
        got = self.primary.dma_read(user_dma_core.DMA_DEVICE_C2H, base, buf, 64)
        if got != 64:
            raise IOError(f"argmax value read at 0x{base:X} returned {got} of 64 bytes")
        # bf16 is just the high 16 bits of an fp32, so widening is exact: append 16 zero
        # bits. Done with struct rather than torch.frombuffer, which warns about tensors
        # over immutable buffers and would need a copy to silence.
        bits = int.from_bytes(buf[off:off + 2], "little") << 16
        return struct.unpack("<f", bits.to_bytes(4, "little"))[0]

    def emit_primary_prologue(self, sw: ShardedWeight, gpr_w: int, gpr_s: int) -> None:
        """Primary: seed its private weight/scale base registers, BEFORE the layer loop.

        The primary cannot reuse the folded body's ``gpr_layer_off``: that advances by
        the main image's ``LAYER_WEIGHT_SIZE``, whereas engine 0's shard lives in its own
        private arena where consecutive layers are ``shard.layer_stride`` apart -- only
        this engine's columns are stored, packed layer after layer. Hence its own pair of
        cursors, advanced by :meth:`emit_primary_layer_advance` once per iteration.
        """
        shard = sw.shard(0)
        self.primary.generate_instruction_add_set(gpr_w, ue_35bit_addr_shifter(shard.weight_addr))
        self.primary.generate_instruction_add_set(gpr_s, ue_35bit_addr_shifter(shard.scale_addr))

    def emit_primary_layer_advance(self, sw: ShardedWeight, gpr_w: int, gpr_s: int) -> None:
        """Primary: step its private cursors to the next layer's block."""
        shard = sw.shard(0)
        self.primary.generate_instruction_add_imm(
            src_reg_idx=gpr_w, immediate_value=ue_35bit_addr_shifter(shard.layer_stride),
            dst_reg_idx=gpr_w)
        self.primary.generate_instruction_add_imm(
            src_reg_idx=gpr_s, immediate_value=ue_35bit_addr_shifter(shard.scale_layer_stride),
            dst_reg_idx=gpr_s)

    def emit_primary_matmat(self, sw: ShardedWeight, a_addr: int, out_addr: int,
                            gpr_w: int, gpr_s: int,
                            gpr_a: Optional[int] = None, gpr_out: Optional[int] = None,
                            gpr_M_reg: Optional[int] = None, gelu: bool = False) -> int:
        """Emit ENGINE 0's block of a sharded quantized matmul into the primary's capture.

        Only engine 0's own columns are computed here; the workers' blocks are emitted by
        :meth:`emit_worker_program`, and the caller owns the barrier
        (:meth:`emit_release_workers` before, :meth:`emit_join_workers` after).

        ``gpr_w`` / ``gpr_s`` are the cursors seeded by :meth:`emit_primary_prologue`.

        Returns engine 0's own FLOPs -- add :meth:`worker_flops` for the whole op.
        """
        shard = sw.shard(0)
        ue = self.primary
        out_slice = out_addr + shard.col_offset * BPE
        kwargs = dict(
            M=1, K=sw.K, N=shard.cols,
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

    def emit_release_workers(self) -> None:
        """Primary, phase 1 of the round: raise the release flag.

        Four-phase handshake, one round::

            master:  SET | work | CHECK_SET(all W) | CLEAR | CHECK_CLEAR(all W)
            worker:  CHECK_SET(0) | work | SET | CHECK_CLEAR(0) | CLEAR

        Every flag EDGE is acknowledged before the next one in that direction can happen,
        so no participant can act on a level left over from the previous round:

          1. M:0->1  "go"            workers wait in CHECK_SET(0)
          2. W:0->1  "I am done"     master waits in CHECK_SET(W)
          3. M:1->0  "round closed"  workers wait in CHECK_CLEAR(0)
          4. W:1->0  "I am re-armed" master waits in CHECK_CLEAR(W)

        The master cannot reach the next round's SET without having seen every worker
        clear (4), so a stale W can never satisfy its CHECK_SET; a worker cannot reach its
        next CHECK_SET without having seen the master clear (3), so a stale M can never
        release it twice. That is a correctness argument, not a timing margin -- it needs
        no delay, and it does not care how uneven the shards are or how little work the
        master has between rounds. Both were load-bearing in the eltwise-delay workaround
        this replaces, and both were why mlp_down sharding failed at 4 and 8 cores.
        """
        if self.num_engines == 1:
            return
        self.primary.generate_instruction_flag_set()                       # 1

    def emit_join_workers(self) -> None:
        """Primary, phases 2-4: wait for all done, close the round, wait for all re-armed."""
        if self.num_engines == 1:
            return
        for idx in self.worker_indices():
            self.primary.generate_instruction_flag_check_set(target_engine_idx=idx)    # 2
        self.primary.generate_instruction_flag_clear()                                 # 3
        for idx in self.worker_indices():
            self.primary.generate_instruction_flag_check_clear(target_engine_idx=idx)  # 4

    def emit_attention(self, ue, engine_idx: int, op: "AttentionOp",
                       gpr_batch: int, gpr_aligned: int, gpr_scale: int,
                       gpr_k: int, gpr_v: int, gpr_q: int, gpr_bias: int,
                       gpr_out: int, gpr_tmp: int) -> int:
        """Emit ONE engine's batch slice of a unified_attention_core.

        Slicing, per dimension:
          Q, OUT  ``[batch, head_dim]``      -> + batch_offset * head_dim * 2, a literal
          BIAS    ``[batch, aligned_seq]``   -> + batch_offset * aligned * 2, and ``aligned``
                                                is only known at RUN time, so the offset is
                                                computed on-device: reg_mul_imm then add_imm.
          K, V                               -> NOT sliced (one KV head, shared by the group)
          SCRATCH                            -> this engine's private buffer

        An engine with zero rows emits nothing -- it still runs the round's handshake, it
        just has no work.
        """
        shard = op.sa.shard(engine_idx)
        if shard.batch_rows == 0:
            return 0
        H, bpe = op.sa.head_dim, BPE
        row_bytes = H * bpe
        ue.generate_instruction_add_set(gpr_batch, shard.batch_rows)
        _const_addr(ue, op.q_addr + shard.batch_offset * row_bytes, gpr_q)
        _const_addr(ue, op.out_addr + shard.batch_offset * row_bytes, gpr_out)
        if shard.batch_offset:
            # bias row stride is the RUNTIME aligned KV length: gpr_bias = base + off*aligned*2
            ue.generate_instruction_reg_mul_imm(
                gpr_tmp, gpr_aligned, ue_35bit_addr_shifter(shard.batch_offset * bpe))
            ue.generate_instruction_add_imm(
                src_reg_idx=gpr_tmp, immediate_value=ue_35bit_addr_shifter(op.bias_addr),
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
        """FLOPs the WORKERS contribute to one attention invocation (engines 1..n-1)."""
        per_row = 2 * 2 * sa.aligned_seq_len * sa.head_dim   # Q@K^T + P@V
        return sum(s.batch_rows * per_row for s in sa.shards[1:])

    def worker_flops(self, sw: ShardedWeight) -> int:
        """FLOPs contributed by engines 1..n-1 for one invocation of ``sw``."""
        return sum(2 * 1 * sw.K * s.cols for s in sw.shards[1:])

    def emit_worker_program(self, rounds, layers: int, tail_rounds=()) -> None:
        """Compile and upload each worker's whole decode program.

        ``rounds`` is a list of ROUNDS; each round is a list of
        ``(ShardedWeight, a_addr, out_addr, gelu)`` ops, and each round is one rendezvous.
        Ops inside a round run back-to-back with no barrier between them -- legal only when
        they stay in this engine's lane, i.e. none of them reads a column another engine
        produced. gate and up qualify (both read pre-MLP-norm, both write their own output
        slice); mlp down does NOT, because its K spans the whole gate*up product, so it
        needs its own round after the master has done the multiply.

        ``tail_rounds`` has the same shape and runs ONCE after the layer loop -- the LM
        head, which is once-per-token and layer-independent, so it uses literal addresses
        and needs no cursor.

        A worker mirrors the primary's folded structure: one captured body, hardware looped
        ``layers`` times. Each round re-arms its flag, waits for the primary's release, runs
        its column block of every op in that round, raises its done flag and burns the skew
        margin. At the end of the layer every op's private weight/scale bases advance by
        that op's OWN packed stride.
        """
        for idx in self.worker_indices():
            ue = self.engines[idx]
            ue.clear_inst_id()
            # A worker's program is re-emitted WHOLE on every compile, and compile_gemma3
            # runs twice under --profile. alloc_isa_reg()'s counter is cumulative, so
            # without this reset the second pass starts high and gpr_batch / gpr_one fall
            # out of the narrow 1..15 window the PBI row-loop trip counts require. The
            # master releases its registers instead; a worker owns its whole stream, so
            # resetting is both simpler and idempotent.
            ue.reset_isa_reg_counter()
            ue.start_capture()
            ue.generate_instruction_flag_clear()

            # gpr_M_reg FIRST: it is the matmul's PBI row-loop trip count, and that field is
            # narrow, so it must land in the low register window (1..15). A worker's counter
            # starts at 1, so allocating it first puts it at r1. The dimension GPR is also
            # what selects quantized_matmat_core's DYNAMIC path -- gpr_*_addr alone is
            # rejected, and the static path could not take a runtime weight base anyway.
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
            self._gpr_aligned[idx] = gpr_aligned

            # One private weight/scale cursor pair per op, across every round: each arena
            # packs only THIS engine's columns, so every op steps by its own shard stride,
            # never by the main image's LAYER_WEIGHT_SIZE.
            cursors = {}
            for rnd in rounds:
                for op in rnd:
                    if isinstance(op, AttentionOp):
                        continue
                    sw = op[0]
                    shard = sw.shard(idx)
                    gpr_w = ue.alloc_isa_reg()
                    gpr_s = ue.alloc_isa_reg()
                    ue.generate_instruction_add_set(gpr_w, ue_35bit_addr_shifter(shard.weight_addr))
                    ue.generate_instruction_add_set(gpr_s, ue_35bit_addr_shifter(shard.scale_addr))
                    cursors[id(sw)] = (shard, gpr_w, gpr_s)

            ue.pad_capture_to_64b_boundary()
            body_word_addr = ue_35bit_addr_shifter(
                ue.get_program_dram_addr() + ue.capture_count * user_dma_core.INSTRUCTION_SIZE_BYTES)

            # --- per-layer body: one rendezvous per round ---
            for rnd in rounds:
                ue.generate_instruction_flag_check_set(target_engine_idx=0)   # 1: wait for go
                for op in rnd:
                    if isinstance(op, AttentionOp):
                        # K/V live in the layer's slice of the shared KV cache: base +
                        # gpr_kv, the worker's own copy of the master's gpr_kv_off.
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
                        OUTPUT_DRAM_ADDR=out_addr + shard.col_offset * BPE,
                        SCALE_DRAM_ADDR=shard.scale_addr,
                        data_type=sw.data_type, gelu_enable=gelu,
                        gpr_M_reg=gpr_one, gpr_b_addr=gpr_w, gpr_scale_addr=gpr_s,
                    )
                ue.generate_instruction_flag_set()                            # 2: done
                ue.generate_instruction_flag_check_clear(target_engine_idx=0)  # 3: round closed
                ue.generate_instruction_flag_clear()                           # 4: re-armed

            if _attn:
                ue.generate_instruction_add_imm(
                    src_reg_idx=gpr_kv,
                    immediate_value=ue_35bit_addr_shifter(_attn[0].kv_layer_stride),
                    dst_reg_idx=gpr_kv)
            for shard, gpr_w, gpr_s in cursors.values():
                ue.generate_instruction_add_imm(
                    src_reg_idx=gpr_w, immediate_value=ue_35bit_addr_shifter(shard.layer_stride),
                    dst_reg_idx=gpr_w)
                ue.generate_instruction_add_imm(
                    src_reg_idx=gpr_s,
                    immediate_value=ue_35bit_addr_shifter(shard.scale_layer_stride),
                    dst_reg_idx=gpr_s)
            ue.generate_instruction_add_dec(reg_idx=gpr_cnt)
            ue.generate_instruction_jump_abs_jnz(body_word_addr, gpr_cnt)

            # --- once-per-token tail rounds (LM head): literal addresses, no cursors ---
            for rnd in tail_rounds:
                ue.generate_instruction_flag_check_set(target_engine_idx=0)   # 1
                for sw, a_addr, out_addr, gelu in rnd:
                    # Writeback ENABLED: global_argmax() has to read these logits back,
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
            self._programs[idx] = addr
            self._body_word_addr[idx] = ue_35bit_addr_shifter(addr)
            # Slot for the per-token preamble, rewritten by start_workers when attention is
            # sharded. Reserved even when it is not, so the ISA accounting is uniform.
            self._preamble[idx] = ue.get_program_dram_addr()
            ue.allocate_program_dram(4 * user_dma_core.INSTRUCTION_SIZE_BYTES)

    @staticmethod
    def _emit_ack_margin(ue) -> None:
        """SHORT delay between a worker's FLAG_SET and its FLAG_CLEAR.

        The worker must hold its done-flag up long enough for the master -- which is
        already spinning in flag_check(this engine) -- to observe it. Clear too early and
        the master waits forever on a 1 it never saw: a HANG.

        Small on purpose. The bound is one poll of an already-spinning checker, not any
        property of the workload, so this does NOT need to scale with core count.
        """
        ue.eltwise_add_core(
            vector_A_sram_start_addr=0x10000,     # URAM_A
            vector_B_sram_start_addr=0x90000,     # URAM_B (add requires different URAMs)
            vector_C_sram_wb_addr=0x10000,
            element_size=ACK_MARGIN_ELEMENTS)

    @staticmethod
    def _emit_skew_margin(ue) -> None:
        """LONG delay between a worker's FLAG_CLEAR and its next FLAG_CHECK(master).

        STAND-IN for the CHECK_ZERO(master) this ISA cannot express. Having cleared, the
        worker loops back to flag_check(0) -- but the master's release is still asserted at
        that moment, because the master only clears once the LAST worker has reported. A
        fast worker would therefore read its own round's release a second time and run the
        next round against activations the master has not produced yet. The proper fix is a
        wait-for-zero on the master's flag (FLAG_MODE encodes only SET/CLEAR/CHECK, and
        CHECK spins on level-1 only).

        Until the hardware grows that op, burn time: the delay only has to exceed the
        completion SKEW between workers, because the master clears at
        max(worker finish) + epsilon.

        CRITICALLY, this sits AFTER the CLEAR. With it before, the worker's done-flag
        stayed asserted for the whole delay, which is exactly the window the master's NEXT
        round samples -- so the same margin that protected the worker made the master read
        a stale 1. That coupled the two bounds into

            worker skew  <  delay  <  master's next-round work

        whose right-hand side shrinks as cores grow (the master's own shard gets smaller),
        which is why 2 cores passed while 4 and 8 failed. Clearing first decouples them:
        this delay now has NO upper bound, and its only cost is worker idle time.

        SRAM-ONLY, deliberately: decode is bandwidth-bound (~500 MB/token), so a delay that
        touched DRAM would steal bandwidth from whatever the master runs concurrently. This
        reads and writes URAM only; its operands are whatever the matmuls left behind and
        its result is discarded -- it is a timer, not a computation.
        """
        ue.eltwise_add_core(
            vector_A_sram_start_addr=0x10000,
            vector_B_sram_start_addr=0x90000,
            vector_C_sram_wb_addr=0x10000,
            element_size=SKEW_MARGIN_ELEMENTS)

    # -- space protection ---------------------------------------------------
    def _check_isa_fits(self, engine_idx: int, addr: int, size_bytes: int) -> None:
        """Refuse to write a worker program that would leave its private ISA slice.

        The ISA slice is only 16 MB and sits directly below the tensor slice, which sits
        directly below the NEXT engine's window. An overrun would not fault -- it would
        silently scribble instructions over a neighbour's weights, so it has to be caught
        at emit time. The program cursor also accumulates across compiles (a --profile run
        compiles twice), which is the realistic way to run out.
        """
        region = self.regions[engine_idx]
        limit = region.isa_base + PRIVATE_ISA_BYTES
        if addr < region.isa_base or addr + size_bytes > limit:
            raise MemoryError(
                f"engine {engine_idx} ISA overflow: program [0x{addr:X}..0x{addr + size_bytes:X}) "
                f"is outside its slice [0x{region.isa_base:X}..0x{limit:X}) "
                f"({PRIVATE_ISA_BYTES // 2**20} MB). Writing it would corrupt the "
                f"{'tensor slice' if addr + size_bytes <= region.tensor_base + PRIVATE_TENSOR_BYTES else 'next engine window'}.")

    def verify_private_space(self, verbose: bool = True) -> None:
        """Check every engine's weight arena and ISA slice are within their window.

        Cheap, host-side, and worth calling after setup: an overflow here is silent
        corruption of a neighbouring engine's memory, not a fault.
        """
        for i, region in enumerate(self.regions):
            used = self._weight_cursor[i] - region.weight_base
            cap = region.weight_limit - region.weight_base
            if self._weight_cursor[i] > region.weight_limit:
                raise MemoryError(
                    f"engine {i} weight arena overflow: {used / 2**20:.1f} MB used of "
                    f"{cap / 2**20:.1f} MB, spilling into its ISA slice at "
                    f"0x{region.isa_base:X}")
            isa_used = self.engines[i].get_program_dram_addr() - region.isa_base
            if i > 0 and not 0 <= isa_used <= PRIVATE_ISA_BYTES:
                raise MemoryError(
                    f"engine {i} ISA cursor 0x{self.engines[i].get_program_dram_addr():X} "
                    f"is outside its {PRIVATE_ISA_BYTES // 2**20} MB slice at "
                    f"0x{region.isa_base:X}")
            if verbose:
                isa_txt = f"{isa_used / 1024:7.1f} KB" if i > 0 else "     (main map)"
                print(f"    core {i}: weights {used / 2**20:6.1f} / {cap / 2**20:.0f} MB"
                      f"   isa {isa_txt} / {PRIVATE_ISA_BYTES // 2**20} MB")

    # -- run ----------------------------------------------------------------
    def start_workers(self, aligned_seq_len: Optional[int] = None) -> None:
        """Launch every worker; they block on the primary's first release.

        With attention sharded the workers need this token's aligned KV length, which
        their once-compiled body cannot know. Pass it and each worker is entered through a
        freshly written 2-instruction preamble -- ``add_set(gpr_aligned, N)`` then a jump
        into the body -- exactly how run_gemma3_decode primes the master.
        """
        for idx in self.worker_indices():
            ue = self.engines[idx]
            gpr_aligned = self._gpr_aligned.get(idx)
            if aligned_seq_len is None or gpr_aligned is None:
                ue.start_execute_from_dram(self._programs[idx])
                continue
            ue.clear_inst_id()
            ue.start_capture()
            ue.generate_instruction_add_set(gpr_aligned, aligned_seq_len)
            ue.generate_instruction_jump_abs(self._body_word_addr[idx])
            ue.stop_capture()
            ue.write_captured_instructions_to_dram(self._preamble[idx])
            ue.clear_capture_buffer()
            ue.start_execute_from_dram(self._preamble[idx])

    def reset_workers(self) -> None:
        """Clear stale flags left set by an aborted run, so the next barrier starts clean.

        A worker that was killed mid-rendezvous leaves its done-flag raised; the next run's
        primary would then sail through its first ``flag_check`` and read a half-written
        output slice. Run this before launching, not after failing.
        """
        for ue in self.workers:
            ue.clear_inst_id()
            ue.start_capture()
            ue.generate_instruction_flag_clear()
            ue.generate_instruction_halt()
            ue.stop_capture()
            addr = ue.get_program_dram_addr()
            ue.write_captured_instructions_to_dram(addr)
            ue.clear_capture_buffer()
            ue.program_execute(addr, timeout=1.0)


# --------------------------------------------------------------------------
# small emit helpers (mirror the folded decoder's const_addr / layer_addr)
# --------------------------------------------------------------------------
def _const_addr(ue, literal: int, scratch_reg: int) -> int:
    ue.generate_instruction_add_set(scratch_reg, ue_35bit_addr_shifter(literal))
    return scratch_reg


def _offset_addr(ue, gpr_off: int, literal: int, scratch_reg: int) -> int:
    ue.generate_instruction_add_imm(src_reg_idx=gpr_off,
                                    immediate_value=ue_35bit_addr_shifter(literal),
                                    dst_reg_idx=scratch_reg)
    return scratch_reg
