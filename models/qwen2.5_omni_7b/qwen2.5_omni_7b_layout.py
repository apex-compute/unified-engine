#!/usr/bin/env python3
"""DRAM layout of the Qwen2.5-Omni-7B runtime, separated from the engine.

The engine (qwen2.5_omni_7b_test.py) and the stage modules (lm / vision /
audio) never touch window arithmetic directly: they ask a ``MemoryLayout`` for
weights, tensors, ISA space and phase marks. ``WindowedLayout`` is the
production strategy -- eight private 1 GiB windows with a shared pool carved
from each window's tail. Another strategy only has to provide the same surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import user_dma_core
from multi_engine_shard import (PRIVATE_ALIGN, PrivateArena, PrivateRegion,
                                require_multicore_dram, tiled_window_bases)

MODEL_NAME = "Qwen2.5-Omni-7B"
REQUIRED_ENGINES = 8
REQUIRED_DRAM_GIB = 8

# ==========================================================================
# THE WINDOW IS ONE HBM SWITCH REGION (Alveo U50 and U55C)
# ==========================================================================
# A window is 1 GiB because that is the granularity the HBM fabric arbitrates
# on, and eight of them tile the 8 GiB device so every core reads its own.
# Which board is underneath decides only WHY 1 GiB is the right number:
#
#   U55C (HW_INFO cores == 12)  one memory controller owns one contiguous
#       1 GiB, and the reordered SAXI wiring puts core i on controller i for
#       i < 8. The window IS the controller. Measured on the PRE-reorder
#       bitstream, where all twelve engines crowded onto MC0-MC3, decode ran at
#       77.3 GFLOPS against 140.1 for the old 512 MiB map; with the ports
#       reordered the same map gives 190.2.
#   U50 (HW_INFO cores == 8)    the contended unit is the 1 GiB four-pseudo-
#       channel switch region, not the 512 MiB controller: two engines in one
#       region still read at the full per-engine rate, three halve it. A 1 GiB
#       window IS one switch region, so this map puts ONE engine in each.
#
# The U50 is where the map looks portable but is not obviously so, because an
# engine owns two 512 MiB segments outright -- one per 4 GiB stack, on its own
# SAXI port -- and its 1 GiB window here is neither of them. That is deliberate.
# Port ownership decides whether a read takes a lateral hop, and at these
# transfer sizes the hop is free: the port-ordered bases and a flat 512 MiB
# stride both measure 85.2 GB/s, the per-engine AXI ceiling. Crowding is what
# costs bandwidth, and a contiguous 1 GiB window per core cannot crowd. Keeping
# the window contiguous is also what keeps the map feasible at all -- the
# vision encoder's ISA slice plus a full private shard set (~712 MiB, see
# OMNI_PRIVATE_RESERVE_BYTES) needs more than a 512 MiB segment offers.
#
# multi_engine_shard.tiled_window_bases() owns both boards' answers and
# validates the result (in range, disjoint, no crowded region); a board it has
# not characterised is refused rather than guessed at.

# The private window geometry this map is built for. It is deliberately NOT
# multi_engine_shard.MULTICORE_WINDOW_BYTES: that constant is 512 MiB and three
# other multi-core models are validated against it. Nor does it track either
# board's constant: it is THIS MAP's geometry, and every number below --
# OMNI_PRIVATE_RESERVE_BYTES, the ISA and tensor slices -- is tuned against
# it. tiled_window_bases() refuses a board whose own window
# differs (the U55C's becomes 2 GiB when its HBM is upgraded), so the map gets
# re-tuned deliberately instead of silently running at the wrong size.
OMNI_WINDOW_BYTES = 0x4000_0000            # 1 GiB per core, 8 GiB total
# Medium-resolution vision has a 47.75 MiB master program, so ISA must retain
# 64 MiB. Place it at the END of each window, with 32 MiB deliberately unused
# immediately before it. This strip separates any overrun from the private
# tensor slice below it; it does not make HBM writes fault or prove the ISA safe.
OMNI_ISA_BYTES = 64 * 2**20
OMNI_ISA_GUARD_BYTES = 32 * 2**20
# 264 MiB, not 64: past a 4096 context, the per-engine attention scratch
# (unified_attention_core's [aligned_seq_len, aligned_seq_len] score buffer,
# quadratic in context -- see lm_attn_scratch's own MemoryError) plus the
# private MLP down/gate/up lanes (see LM_MLP_*_PER_ENGINE in
# qwen2.5_omni_7b_lm.py) no longer fit in 64 MiB. 64 MiB was exact for a 4096
# context (~62 MiB on core 0); 8192 needs ~262 MiB, so this carries a small
# margin over that, not the old ceiling.
OMNI_PRIVATE_TENSOR_BYTES = 264 * 2**20

# The measured IF4 projection + LM-head footprint is 447.8 MiB/core. Reserve
# 464 MiB for private shards before lending anything to the shared pool -- the
# measured footprint plus 16 MiB. It was 512, which left 64 MiB per core
# reserved-but-never-used: 513 MiB across the board that the shared pool could
# not touch. The footprint is deterministic (a fixed model at a fixed eight
# engines; vision/audio weights are PARAMS, not private shards), and overshoot
# fails loudly in alloc_shared rather than corrupting, so the slack buys
# nothing. The remaining 200 MiB of the 664 MiB weight arena is shared
# capacity; it grows only within [base+0x1D000000, base+0x29800000). (This
# band shrank from 400 MiB when OMNI_PRIVATE_TENSOR_BYTES grew from 64 to 264
# MiB above -- the two budgets share the same 1 GiB window and trade off
# directly.)
#
# THE BAND, NOT THE TOTAL, IS WHAT BINDS. A shared object must fit one
# contiguous per-window gap. GATE/UP used to be the objects that mattered here
# (296 MiB each at an 8192 context) but are now private per-engine lanes (see
# OMNI_PRIVATE_TENSOR_BYTES above), so the largest remaining shared object at
# 8192 is the qkv/attn/mlp overlay or LM_BIAS, 128 MiB each -- comfortably
# inside 200 MiB.
OMNI_PRIVATE_RESERVE_BYTES = 464 * 2**20

# FPGA speech keeps the Thinker's private decode shards resident while adding
# 88.3 MiB of Talker column shards per busy worker. A 544 MiB reserve covers
# both measured footprints (447.8 + 88.3) with 7.9 MiB left. The Talker
# selection also narrows the Thinker context to 6144: its worst private tensor
# requirement then falls below 184 MiB, preserving a 200 MiB shared band and
# the 32 MiB ISA guard. Host speech and text-only runs retain the 8192 map.
FPGA_SPEECH_CONTEXT_SIZE = 6144
FPGA_SPEECH_PRIVATE_TENSOR_BYTES = 184 * 2**20
FPGA_SPEECH_PRIVATE_RESERVE_BYTES = 544 * 2**20

# THERE USED TO BE A SECOND UNSCATTERABLE OBJECT HERE: a 280 MiB "dedicated
# extent" (OMNI_LM_HEAD_BYTES) holding one contiguous, unsharded copy of the
# untied LM head, loaded via a normal device DMA. It turned out to have
# exactly one reader: _ensure_decode_shards_impl's shard_quantized_weight
# call, which immediately re-copied it (card -> host -> card, on top of the
# blob's own host -> card load) into the SAME eight private per-engine column
# shards already budgeted above as "lm_head shard 34.5 MiB". Nothing else
# ever read the unsharded copy -- this model always runs eight engines
# (REQUIRED_ENGINES), so decode's sharded head path is always taken and the
# one call site that read the whole-blob fallback is unreachable. Removed:
# lm_head is now read directly from the host file and column-sharded straight
# into the private shards it always ended up in (shard_quantized_weight_from_
# bytes), one host->card DMA per shard instead of three device round trips
# for the whole weight. That freed 280 MiB from whichever ONE core alloc_
# shared had picked to host it (always core 0, the only core that previously
# had ZERO shared-pool slack) -- room now spent on the OMNI_ISA_BYTES growth
# above, uniformly across every core.


class MemoryLayout:
    """The surface the engine and stage modules rely on.

    Concrete layouts subclass PrivateArena (alloc_weights / alloc_tensor /
    alloc_shared / alloc_shared_up / check_isa_fits / region / isa_base / ...)
    and add the lifecycle below.
    """

    kind = "abstract"
    # True when starting one stage's weights destroys another stage's (the
    # windowed map time-shares one pool); False when every stage stays resident.
    evicts_weights = True
    # True when every stage's program stays in its own ISA range, so programs
    # are loaded once rather than reloaded before each stage runs.
    resident_programs = False

    def begin_weight_stage(self, stage: str) -> None: ...
    def validate_config(self, hw: dict, tensor_bytes: int) -> None: ...
    def program_identity(self) -> dict: ...
    def release_weight_phase(self) -> int: ...
    def release_tensor_phase(self) -> None: ...
    def tensor_phase_mark(self): ...
    def tensor_phase_restore(self, mark) -> None: ...


class WindowedLayout(PrivateArena, MemoryLayout):
    """Eight private 1 GiB windows tiling the device, one per engine.

    Inside each window, low to high: private weights, a shared pool (weights
    and tensors of the stage currently running), the private tensor slice, an
    untouched guard, then the ISA slice. Vision, audio and the LM time-share
    the pool through shared_mark()/shared_release().
    """

    kind = "windowed"
    evicts_weights = True

    def begin_weight_stage(self, stage: str) -> None:
        """The windowed pool is shared by all stages; nothing to select."""

    def __init__(self, multi_core, speech: bool):
        require_multicore_dram(multi_core, MODEL_NAME)
        bases = tiled_window_bases(REQUIRED_ENGINES, OMNI_WINDOW_BYTES, MODEL_NAME)
        self.private_tensor_bytes = (FPGA_SPEECH_PRIVATE_TENSOR_BYTES if speech
                                     else OMNI_PRIVATE_TENSOR_BYTES)
        self.private_reserve_bytes = (FPGA_SPEECH_PRIVATE_RESERVE_BYTES if speech
                                      else OMNI_PRIVATE_RESERVE_BYTES)
        self.window_bytes = OMNI_WINDOW_BYTES
        # Top of the highest window: the whole device on the 8 GiB image, a
        # bound on the 16 GiB one where the windows are a permutation.
        self.device_end = max(bases) + OMNI_WINDOW_BYTES
        # Built from the board's bases, not base-plus-stride.
        super().__init__(
            REQUIRED_ENGINES,
            windows=[(b, OMNI_WINDOW_BYTES) for b in bases],
            isa_bytes=OMNI_ISA_BYTES,
            tensor_bytes=self.private_tensor_bytes,
            isa_guard_bytes=OMNI_ISA_GUARD_BYTES,
            verbose=True,
        )
        if self.stride != OMNI_WINDOW_BYTES:
            raise AssertionError(
                f"private windows are 0x{self.stride:X}, not the "
                f"0x{OMNI_WINDOW_BYTES:X} this map is built for")
        actual = [self.window_base(i) for i in range(REQUIRED_ENGINES)]
        if actual != bases:
            raise AssertionError(
                "private windows do not match this board's map: "
                f"{[hex(a) for a in actual]} against {[hex(b) for b in bases]}")
        if self.device_end > user_dma_core.AVAILABLE_DRAM_SIZE_GB * 2**30:
            raise AssertionError(
                f"{REQUIRED_ENGINES} x {OMNI_WINDOW_BYTES // 2**20} MiB windows "
                f"reach 0x{self.device_end:X}, past the "
                f"{user_dma_core.AVAILABLE_DRAM_SIZE_GB} GiB HW_INFO reports")
        # Private space is claimed before any shared byte is lent.
        self.reserve_private(self.private_reserve_bytes)
        self.shared_capacity = sum(self.shared_free())
        self.tensor_capacity = self.shared_capacity
        self._weight_phase_mark = self.shared_mark()
        self._tensor_phase_shared_mark = self.shared_up_mark()
        self._tensor_phase_private_mark = self.tensor_mark()

    # -- phases ----------------------------------------------------------
    def release_weight_phase(self) -> int:
        """Hand the previous stage's shared weights back; returns bytes."""
        return self.shared_release(self._weight_phase_mark)

    def release_tensor_phase(self) -> None:
        self.shared_up_release(self._tensor_phase_shared_mark)
        self.tensor_release(self._tensor_phase_private_mark)

    def tensor_phase_mark(self):
        return (self.shared_up_mark(), self.tensor_mark())

    def tensor_phase_restore(self, mark) -> None:
        shared_mark, private_mark = mark
        self.shared_up_release(shared_mark)
        self.tensor_release(private_mark)

    # -- contract with config and cached programs ------------------------
    def validate_config(self, hw: dict, tensor_bytes: int | None = None) -> None:
        expected = {
            "required_engines": REQUIRED_ENGINES,
            "required_dram_gib": REQUIRED_DRAM_GIB,
            "private_arena_base": 0,
            "private_arena_bytes": REQUIRED_ENGINES * OMNI_WINDOW_BYTES,
            "private_window_bytes": OMNI_WINDOW_BYTES,
            "private_isa_bytes": OMNI_ISA_BYTES,
            "private_isa_guard_bytes": OMNI_ISA_GUARD_BYTES,
            "private_tensor_bytes": OMNI_PRIVATE_TENSOR_BYTES,
            "private_weight_reserve_bytes": OMNI_PRIVATE_RESERVE_BYTES,
            # The DRAM this map CLAIMS (eight windows), not device_end: on the
            # 16 GiB image the windows are eight of sixteen regions.
            "dram_limit": REQUIRED_ENGINES * OMNI_WINDOW_BYTES,
        }
        for name, wanted in expected.items():
            raw = hw.get(name)
            actual = int(raw, 0) if isinstance(raw, str) else int(raw)
            if actual != wanted:
                raise ValueError(
                    f"config hardware.{name}={raw!r}, expected 0x{wanted:X}")
        if self.stride != int(hw["private_window_bytes"], 0):
            raise AssertionError("PrivateArena did not produce 1-GiB windows")
        expected_weight_bytes = (OMNI_WINDOW_BYTES - OMNI_ISA_BYTES
                                 - OMNI_ISA_GUARD_BYTES
                                 - self.private_tensor_bytes)
        if self.weight_bytes() != expected_weight_bytes:
            raise AssertionError(
                f"each engine must have {expected_weight_bytes // 2**20} MiB for its "
                f"private shards and the shared pool, got "
                f"{self.weight_bytes() // 2**20} MiB")
        expected_tensor_base = expected_weight_bytes
        for i in range(REQUIRED_ENGINES):
            region = self.region(i)
            base = region.base
            if not (region.weight_limit == base + expected_tensor_base
                    and region.tensor_base == base + expected_tensor_base
                    and region.isa_base == base + 0x3C00_0000
                    and self.isa_limit(i) == base + OMNI_WINDOW_BYTES):
                raise AssertionError(f"core {i} is not in the guarded Omni map")

    def program_identity(self, params_base: int, params_limit: int,
                         tensor_base: int, tensor_limit: int) -> dict:
        """The dram_map recorded in the cached programs' identity."""
        return {
            "private_base": 0,
            "params_base": params_base,
            "params_limit": params_limit,
            "tensor_base": tensor_base,
            "tensor_limit": tensor_limit,
            "master_isa_base": self.isa_base(0),
            "worker_isa_base": self.isa_limit(0),
            # Per engine, not base + i * stride: on a board map the windows
            # are a permutation and no stride reaches them.
            "worker_isa_bases": [self.isa_base(i) for i in range(REQUIRED_ENGINES)],
            "dram_end": self.device_end,
        }


MIB = 2**20


def _align_up(n: int, a: int = PRIVATE_ALIGN) -> int:
    return (n + a - 1) // a * a


# WEIGHTS FIRST. Every stage's weights stay resident, so they are sized before
# anything else and are never traded away; instruction images come next; the
# tensor zones take whatever DRAM is left, and that remainder decides the
# longest context the LM can be built for.
#
# Sizes are the measured peaks (vision 345.8, audio 332.6, the LM's shared bf16
# 0.6, Token2Wav 579.8; Thinker private shards 447.8, plus the Talker's 88.3 =
# 536.1) rounded up a few MiB. Everything is carved to 1 MiB, not the windowed
# map's 16.
FLAT_WEIGHT_POOLS_MIB = {"vision": 352, "audio": 340, "lm": 4, "t2w": 584}
FLAT_PRIVATE_RESERVE_MIB = {False: 450, True: 538}          # keyed by talker
FLAT_ISA_GUARD_BYTES = 16 * MIB
FLAT_SLACK_MIB = 8
FLAT_MAX_CONTEXT = 8192
FLAT_CONTEXT_STEP = 256

# Per-resolution vision costs. Weights do not depend on the resolution; the
# encoder's program, its activations and its attention scratch do.
# shared/private are MiB of tensor zone, isa the master program slice.
FLAT_VISION_COST = {
    "small": {"shared": 64, "private": 4, "isa": 16},
    "medium": {"shared": 672, "private": 40, "isa": 52},
}
FLAT_AUDIO_COST = {"shared": 32, "private": 4, "isa": 3}
# Master/worker program images per engine, in MiB, for the stages whose size
# does not depend on the request. Each stage keeps its own range (nothing is
# reloaded between stages) and is followed by a gap that the runtime's small
# dynamic writes (flag-clear programs, decode preambles) land in.
FLAT_LM_ISA_MIB = 10                     # prefill + decode + decode launch tables
FLAT_TALKER_ISA_MIB = 4
FLAT_ISA_GAP_MIB = 1
FLAT_T2W_ISA_MIB = 20                    # 16.6 MiB per engine measured
FLAT_T2W_ACTIVATION_MIB = 384            # 373.7 MiB pool peak at 300 codec tokens

# The LM's tensor cost as a function of the context it is built for. These
# are the formulas lm_tensor_init uses (KV cache plus per-token planes, and the
# per-engine scratch whose attention term is quadratic), checked against two
# measured points: 6144 -> 759.6 MiB shared / 172.6 MiB private and
# 8192 -> 988.6 / 262.
LM_HIDDEN = 3584
LM_MLP_LANE = 18944 // 8
LM_ATTN_HEAD_DIM = 128


def lm_shared_tensor_bytes(context: int) -> int:
    per_token = 117 * 1024 + 700        # KV (56 KiB) + the [P, *] planes
    # The two measured points are linear to within a few MiB; the margin covers
    # the 64-row plane alignment between them.
    return int((per_token * context + 72.6 * MIB) * 1.03) + 4 * MIB


def lm_private_tensor_bytes(context: int) -> int:
    p = (context + 63) // 64 * 64
    scratch = ((LM_ATTN_HEAD_DIM + p) * p + p * LM_ATTN_HEAD_DIM) * 2
    return scratch + p * LM_HIDDEN * 2 + 2 * p * LM_MLP_LANE * 2 + 2 * MIB


@dataclass
class FlatPlan:
    stages: frozenset
    vision_res: str
    max_context: int
    private_reserve: int
    pools: dict
    shared_tensor: int
    private_tensor: int
    isa_stage: int
    isa_t2w: int
    guard: int
    total: int

    def lines(self) -> list[str]:
        return [f"  [plan] stages {sorted(self.stages)}, vision {self.vision_res}: "
                f"max context {self.max_context}, {self.total / MIB:.0f} of "
                f"{REQUIRED_DRAM_GIB * 1024} MiB"]


def plan_flat(stages, vision_res: str = "small", max_context: int | None = None,
              device_bytes: int = REQUIRED_DRAM_GIB * 2**30) -> FlatPlan:
    """Fix the weights and instruction images, then give the rest to the context."""
    stages = frozenset(stages) | {"lm"}
    n = REQUIRED_ENGINES
    priv = FLAT_PRIVATE_RESERVE_MIB["talker" in stages] * MIB
    pools = {k: v * MIB for k, v in FLAT_WEIGHT_POOLS_MIB.items()
             if k == "lm" or k in stages}
    vis = FLAT_VISION_COST[vision_res] if "vision" in stages else None
    # Every stage's program lives in its own range of the slice.
    isa_stage = (FLAT_LM_ISA_MIB
                 + (vis["isa"] if vis else 0)
                 + (FLAT_AUDIO_COST["isa"] if "audio" in stages else 0)
                 + (FLAT_TALKER_ISA_MIB if "talker" in stages else 0)
                 + FLAT_ISA_GAP_MIB * (1 + sum(s in stages for s in
                                               ("vision", "audio", "talker")))) * MIB
    isa_t2w = FLAT_T2W_ISA_MIB * MIB if "t2w" in stages else 0
    fixed = (n * priv + sum(pools.values()) + FLAT_ISA_GUARD_BYTES
             + n * (isa_stage + isa_t2w))
    budget = device_bytes - fixed - FLAT_SLACK_MIB * MIB
    others_shared = max(
        [0, vis["shared"] * MIB if vis else 0,
         FLAT_AUDIO_COST["shared"] * MIB if "audio" in stages else 0,
         FLAT_T2W_ACTIVATION_MIB * MIB if "t2w" in stages else 0])
    others_private = max(
        [0, vis["private"] * MIB if vis else 0,
         FLAT_AUDIO_COST["private"] * MIB if "audio" in stages else 0])

    def cost(c: int) -> tuple[int, int]:
        return (_align_up(max(lm_shared_tensor_bytes(c), others_shared), MIB),
                _align_up(max(lm_private_tensor_bytes(c), others_private), MIB))

    cap = min(max_context or FLAT_MAX_CONTEXT, FLAT_MAX_CONTEXT)
    chosen = None
    for c in range(cap // FLAT_CONTEXT_STEP * FLAT_CONTEXT_STEP, 0, -FLAT_CONTEXT_STEP):
        s, pr = cost(c)
        if s + n * pr <= budget:
            chosen = c
            break
    if chosen is None:
        raise MemoryError(
            f"the weights and instruction images of {sorted(stages)} take "
            f"{fixed / MIB:.0f} MiB of {device_bytes // MIB}; no context fits "
            f"in the remaining {budget / MIB:.0f} MiB")
    if max_context and chosen < max_context:
        raise MemoryError(
            f"--max-context {max_context} does not fit: after the weights the "
            f"largest context is {chosen}")
    s, pr = cost(chosen)
    return FlatPlan(stages=stages, vision_res=vision_res, max_context=chosen,
                    private_reserve=priv, pools=pools, shared_tensor=s,
                    private_tensor=pr, isa_stage=isa_stage, isa_t2w=isa_t2w,
                    guard=FLAT_ISA_GUARD_BYTES,
                    total=fixed + s + n * pr)


class FlatLayout(PrivateArena, MemoryLayout):
    """One consecutive 8 GiB map; every stage's weights stay resident.

    Low to high, with no per-engine window arithmetic:

        private weights   engine 0 .. engine 7          (Thinker + Talker shards)
        shared weights    vision | audio | lm           (one persistent pool each)
        shared tensors    one zone every stage re-carves (outputs go to the host)
        private tensors   engine 0 .. engine 7          (also re-carved per stage)
        ISA guard         untouched
        ISA               engine 0 .. engine 7

    Weights are never evicted: starting a stage selects that stage's own pool
    (begin_weight_stage) and re-initialising it only rewinds that pool. Tensors
    alias across stages exactly as in the windowed map. No attempt is made to
    keep concurrent engines on separate HBM windows; congestion is accepted in
    exchange for fitting everything.
    """

    kind = "flat"
    evicts_weights = False
    resident_programs = True

    def __init__(self, multi_core, plan: FlatPlan):
        require_multicore_dram(multi_core, MODEL_NAME)
        total = user_dma_core.AVAILABLE_DRAM_SIZE_GB * 2**30
        if total < REQUIRED_DRAM_GIB * 2**30:
            raise AssertionError(f"flat map needs {REQUIRED_DRAM_GIB} GiB, "
                                 f"HW_INFO reports {total // 2**30}")
        self.plan = plan
        self.max_context = plan.max_context
        self.private_tensor_bytes = plan.private_tensor
        self.private_reserve_bytes = plan.private_reserve
        self.window_bytes = 0
        self.device_end = REQUIRED_DRAM_GIB * 2**30
        token2wav = "t2w" in plan.stages
        self.token2wav = token2wav
        self._isa_stage_bytes = plan.isa_stage
        self._isa_slice = plan.isa_stage + plan.isa_t2w
        # PrivateArena wants equal windows to set its attributes up; the real
        # geometry replaces everything it derived from them below.
        super().__init__(
            REQUIRED_ENGINES,
            windows=[(i * OMNI_WINDOW_BYTES, OMNI_WINDOW_BYTES)
                     for i in range(REQUIRED_ENGINES)],
            isa_bytes=self._isa_stage_bytes,
            tensor_bytes=self.private_tensor_bytes,
            isa_guard_bytes=plan.guard,
            verbose=False,
        )
        n = REQUIRED_ENGINES
        priv = self.private_reserve_bytes
        self._priv_w = priv
        cursor = 0
        weight_bases = []
        for _ in range(n):
            weight_bases.append(cursor)
            cursor += priv
        self._pools: dict[str, list[int]] = {}      # name -> [base, limit, cursor]
        for name, size in plan.pools.items():
            self._pools[name] = [cursor, cursor + size, cursor + size]
            cursor += size
        self._pool_marks: dict[str, int] = {}
        self._pool_allocs: dict[str, list[dict]] = {k: [] for k in self._pools}
        self._current_pool = "lm"
        self.shared_weight_end = cursor
        self.shared_tensor_base = cursor
        self.tensor_capacity = plan.shared_tensor
        cursor += self.tensor_capacity
        tensor_bases = []
        for _ in range(n):
            tensor_bases.append(cursor)
            cursor += self.private_tensor_bytes
        guard_base = cursor
        isa_base = guard_base + plan.guard
        isa_end = isa_base + n * self._isa_slice
        if isa_end > self.device_end:
            raise AssertionError(
                f"flat map needs 0x{isa_end:X} bytes ({isa_end / MIB:.0f} MiB), "
                f"the device has {self.device_end / MIB:.0f} MiB; "
                f"over by {(isa_end - self.device_end) / MIB:.0f} MiB")
        self.regions = [
            PrivateRegion(engine_idx=i, base=weight_bases[i],
                          weight_base=weight_bases[i],
                          weight_limit=weight_bases[i] + priv,
                          isa_base=isa_base + i * self._isa_slice,
                          tensor_base=tensor_bases[i])
            for i in range(n)]
        self.stride = self._isa_slice           # spacing of the per-engine ISA slices
        self.windows = None
        self._weight_cursor = [r.weight_base for r in self.regions]
        self._weight_allocs = []
        self._tensor_cursor = [r.tensor_base for r in self.regions]
        self._shared_cursor = [r.weight_limit for r in self.regions]
        self._shared_allocs = []
        self._private_reserve = [priv] * n
        self._shared_up_cursor = [self.shared_tensor_base] * n
        self._shared_up_allocs = []
        self._tensor_zone_cursor = self.shared_tensor_base
        self._peak_zone = 0
        self._peak_private_tensor = [0] * n
        self._tensor_zone_limit = self.shared_tensor_base + self.tensor_capacity
        self.shared_capacity = sum(p[1] - p[0] for p in self._pools.values())
        self.map_end = isa_end
        self._tensor_phase_private_mark = self.tensor_mark()
        self._tensor_phase_shared_mark = self.shared_up_mark()

    # -- peaks, so the budget is sized from measurements -----------------------
    def alloc_tensor(self, engine_idx: int, size_bytes: int, what: str) -> int:
        addr = super().alloc_tensor(engine_idx, size_bytes, what)
        used = self._tensor_cursor[engine_idx] - self.regions[engine_idx].tensor_base
        self._peak_private_tensor[engine_idx] = max(
            self._peak_private_tensor[engine_idx], used)
        return addr

    def peak_lines(self) -> list[str]:
        mib = MIB
        pools = ", ".join(f"{k} {(v[1] - v[2]) / mib:.1f}/{(v[1] - v[0]) / mib:.0f}"
                          for k, v in self._pools.items())
        priv = self.usage()
        return [
            "  [map] peaks (used / budget MiB): "
            f"private weights {max(priv) / mib:.1f}/{self._priv_w / mib:.0f}, "
            f"shared tensors {self._peak_zone / mib:.1f}/{self.tensor_capacity / mib:.0f}, "
            f"private tensors {max(self._peak_private_tensor) / mib:.1f}/"
            f"{self.private_tensor_bytes / mib:.0f}, weight pools {pools}"]

    # -- sizes the engine and reports ask for ------------------------------
    def weight_bytes(self) -> int:
        return self._priv_w

    def window_base(self, engine_idx: int) -> int:
        return self.regions[engine_idx].weight_base

    def reserve_private(self, bytes_per_engine: int) -> None:
        if bytes_per_engine > self._priv_w:
            raise ValueError(
                f"private reserve {bytes_per_engine / MIB:.1f} MiB exceeds the "
                f"{self._priv_w / MIB:.0f} MiB private segment")

    # -- shared weights: one persistent pool per stage ----------------------
    def begin_weight_stage(self, stage: str) -> None:
        if stage not in self._pools:
            raise KeyError(f"no weight pool for stage {stage!r}")
        self._current_pool = stage

    def shared_free(self) -> list[int]:
        base, limit, cur = self._pools[self._current_pool]
        free = [cur - base] + [0] * (self.num_engines - 1)
        return free

    def alloc_shared(self, size_bytes: int, what: str, align: int = 128,
                     engine_idx=None) -> int:
        if size_bytes <= 0:
            raise ValueError(f"{what}: shared allocation must be positive")
        pool = self._pools[self._current_pool]
        addr = (pool[2] - size_bytes) & ~(align - 1)
        if addr < pool[0]:
            raise MemoryError(
                f"{what}: {size_bytes / MIB:.2f} MiB does not fit the "
                f"{self._current_pool} weight pool "
                f"({(pool[1] - pool[0]) / MIB:.0f} MiB, "
                f"{(pool[2] - pool[0]) / MIB:.2f} MiB still free)")
        pool[2] = addr
        self._pool_allocs[self._current_pool].append(
            {"engine": 0, "base": addr, "size": size_bytes, "what": what})
        return addr

    def shared_mark(self) -> tuple:
        return (self._current_pool, self._pools[self._current_pool][2])

    def shared_release(self, mark: tuple) -> int:
        name, cur = mark
        pool = self._pools[name]
        reclaimed = cur - pool[2]
        pool[2] = cur
        if cur == pool[1]:
            self._pool_allocs[name].clear()
        return reclaimed

    def release_weight_phase(self) -> int:
        """Rewind only the stage being (re)initialised."""
        pool = self._pools[self._current_pool]
        reclaimed = pool[1] - pool[2]
        pool[2] = pool[1]
        self._pool_allocs[self._current_pool].clear()
        return reclaimed

    def shared_usage(self) -> list[int]:
        used = sum(p[1] - p[2] for p in self._pools.values())
        return [used] + [0] * (self.num_engines - 1)

    def shared_allocations(self) -> list[dict]:
        return [a for allocs in self._pool_allocs.values() for a in allocs]

    # -- shared tensors: one zone, aliased across stages --------------------
    def alloc_shared_up(self, size_bytes: int, what: str, align: int = 128,
                        engine_idx=None) -> int:
        if size_bytes <= 0:
            raise ValueError(f"{what}: allocation must be positive")
        addr = (self._tensor_zone_cursor + align - 1) & ~(align - 1)
        if addr + size_bytes > self._tensor_zone_limit:
            raise MemoryError(
                f"{what}: {size_bytes / MIB:.2f} MiB does not fit the shared "
                f"tensor zone ({self.tensor_capacity / MIB:.0f} MiB, "
                f"{(self._tensor_zone_limit - self._tensor_zone_cursor) / MIB:.2f} "
                f"MiB still free)")
        self._tensor_zone_cursor = addr + size_bytes
        self._peak_zone = max(self._peak_zone, addr + size_bytes - self.shared_tensor_base)
        self._shared_up_allocs.append(
            {"engine": 0, "base": addr, "size": size_bytes, "what": what})
        return addr

    def shared_up_mark(self) -> tuple:
        return (self._tensor_zone_cursor, len(self._shared_up_allocs))

    def shared_up_release(self, mark: tuple) -> int:
        cur, count = mark
        if cur > self._tensor_zone_cursor:
            raise ValueError("shared_up_release: marks release in reverse order only")
        reclaimed = self._tensor_zone_cursor - cur
        self._tensor_zone_cursor = cur
        del self._shared_up_allocs[count:]
        return reclaimed

    def shared_up_usage(self) -> list[int]:
        return [self._tensor_zone_cursor - self.shared_tensor_base] \
            + [0] * (self.num_engines - 1)

    # -- phases --------------------------------------------------------------
    def release_tensor_phase(self) -> None:
        self.shared_up_release(self._tensor_phase_shared_mark)
        self.tensor_release(self._tensor_phase_private_mark)

    def tensor_phase_mark(self):
        return (self.shared_up_mark(), self.tensor_mark())

    def tensor_phase_restore(self, mark) -> None:
        shared_mark, private_mark = mark
        self.shared_up_release(shared_mark)
        self.tensor_release(private_mark)

    # -- Token2Wav's place in the map ------------------------------------------
    def t2w_memory_map(self) -> dict:
        """Where Token2Wav lives: its own weight pool and ISA tail, plus the
        tensor regions every other stage has already finished with."""
        if not self.token2wav:
            raise RuntimeError("this flat map was built without Token2Wav")
        lo, hi, _ = self._pools["t2w"]
        # The pool is staged top-down by alloc_shared; Token2Wav stages bottom-up
        # through UnifiedEngine, so it takes the whole pool as one extent.
        self._pools["t2w"][2] = lo
        const_off = 96 * MIB              # per-engine scratch stays below this
        return {
            "weights_base": lo,
            "weights_limit": hi,
            "pool_segments": [(self.shared_tensor_base, self._tensor_zone_limit)],
            "programs": [(r.isa_base + self._isa_stage_bytes, r.isa_base + self._isa_slice)
                         for r in self.regions],
            "scratch": [r.tensor_base for r in self.regions],
            "consts": [r.tensor_base + const_off for r in self.regions],
        }

    def tensor_phase_set(self, mark) -> None:
        """Move the tensor cursors forward to a mark taken earlier (stage re-entry)."""
        (zone_cursor, count), private = mark
        self._tensor_zone_cursor = zone_cursor
        self._tensor_cursor = list(private)

    # -- report ----------------------------------------------------------------
    def describe(self) -> str:
        lines = [f"  Flat map: {self.map_end / MIB:.0f} of "
                 f"{self.device_end / MIB:.0f} MiB, one consecutive range:"]
        r0, r7 = self.regions[0], self.regions[-1]
        lines.append(f"    private weights  0x{r0.weight_base:09X} .. "
                     f"0x{r7.weight_limit:09X}  8 x {self._priv_w / MIB:.0f} MiB")
        for name, (lo, hi, _) in self._pools.items():
            lines.append(f"    {name:<8} weights 0x{lo:09X} .. 0x{hi:09X}  "
                         f"{(hi - lo) / MIB:.0f} MiB")
        lines.append(f"    shared tensors   0x{self.shared_tensor_base:09X} .. "
                     f"0x{self._tensor_zone_limit:09X}  "
                     f"{self.tensor_capacity / MIB:.0f} MiB")
        lines.append(f"    private tensors  0x{r0.tensor_base:09X} .. "
                     f"+8 x {self.private_tensor_bytes / MIB:.0f} MiB")
        lines.append(f"    ISA              0x{r0.isa_base:09X} .. "
                     f"0x{r7.isa_base + self._isa_slice:09X}  8 x "
                     f"{self._isa_slice / MIB:.0f} MiB (guard "
                     f"{self.plan.guard / MIB:.0f} MiB below)")
        return "\n".join(lines)

    def validate_config(self, hw: dict, tensor_bytes: int | None = None) -> None:
        for name, wanted in (("required_engines", REQUIRED_ENGINES),
                             ("required_dram_gib", REQUIRED_DRAM_GIB)):
            raw = hw.get(name)
            actual = int(raw, 0) if isinstance(raw, str) else int(raw)
            if actual != wanted:
                raise ValueError(f"config hardware.{name}={raw!r}, expected {wanted}")

    def program_identity(self, params_base: int, params_limit: int,
                         tensor_base: int, tensor_limit: int) -> dict:
        return {
            "layout": "flat",
            "private_weight_bases": [r.weight_base for r in self.regions],
            "weight_pools": {k: [v[0], v[1]] for k, v in self._pools.items()},
            "shared_tensor_zone": [self.shared_tensor_base, self._tensor_zone_limit],
            "private_tensor_bases": [r.tensor_base for r in self.regions],
            "master_isa_base": self.isa_base(0),
            "worker_isa_bases": [self.isa_base(i) for i in range(REQUIRED_ENGINES)],
            "isa_bytes": self._isa_slice,
            "isa_stage_bytes": self._isa_stage_bytes,
            "dram_end": self.device_end,
        }
