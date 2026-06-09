"""selector.py — high-level model-porting wrapper for the unified-engine accelerator.

PURPOSE
-------
A single library people import to port a new model with the *minimum* hand-written
code. The selector absorbs every low-level concern — quantization choice, DRAM
partitioning, mask/bias generation, mode selection, hang-avoidance — so a model
author only describes the math at its original (un-padded) dimensions and the tool
does the rest.

QUANT IS THE USER'S CHOICE — THE SELECTOR GUIDES, IT DOES NOT DECIDE
--------------------------------------------------------------------
Quant mode selection stays on the USER's side. They pick precision by choosing which
helpers to call — the bf16 helpers vs. the quantized-method helpers — when they
declare their weights. The selector does NOT silently re-quantize for them.

The feedback loop is driven by `__init__`: when the user writes their model class,
the `__init__` (config pull + weight declaration + DRAM partitioning) is *compiled*
to lay out params/activations/instructions in the 4GB budget. Simply passing a valid
`__init__` gets them a working result — that is the success signal. If `__init__`
does NOT fit, it does not hang or silently truncate: it throws a clear DRAM error
that prompts the user to switch the offending tensors to the quantized-method
helpers (e.g. "params region over budget by 180MB — use the q4/fp4 weight helpers
for the LM blocks"). So: user passes init -> result; init overflows -> actionable
prompt to quantize. The decision remains theirs; the tool only tells them when and
where bf16 won't fit.

WHAT THE SELECTOR DOES
----------------------
1. READ THE MODEL FROM ITS HF CONFIG.
   Given a HuggingFace link (or repo id), pull the model's `config.json` directly,
   read its architecture (hidden sizes, layer count, heads, vocab, intermediate),
   and compute the total parameter footprint so the budget math in `__init__` is
   automatic.

2. SIZE THE bf16 / QUANT BUDGET (REPORT, DON'T OVERRIDE).
   From the hidden sizes and the calculated params size, compute how much of the
   model can fit in bf16 within the params partition and report it. If the user's
   chosen precision mix overflows, raise the DRAM error above pointing at the
   specific tensors to move to quantized helpers. The selector advises; the user's
   helper choice is what actually sets each tensor's precision.

3. AUTO-SCHEDULE EVERYTHING NEEDED TO EXECUTE.
   Bias/causal-mask generation, RoPE table generation, identity/zero allocations,
   and ALL DRAM management (address assignment + liveness reuse) for back-to-back
   operations. The author never names a buffer or computes an address.

4. ROUTES ARE FIXED ONCE, IN ONE PLACE.
   When a particular execution route is locked in (e.g. the PBI/legacy mode mix for
   prefill vs decode, the op ordering, the fused-op flags), it is maintained here in
   selector.py. The high-level model code reflects that route automatically — no
   per-model copies of the routing decisions.

5. AUTO-HANDLE STRUCTURES THAT HANG THE DEVICE.
   Detect code shapes that the hardware cannot execute or that overflow a budget,
   and fail loudly with an actionable message instead of hanging. Examples:
     - out of DRAM budget -> "utilizing too much DRAM budget, try the quantized-mode
       helpers" (with the offending region + overflow amount).
     - too many advancing PBI pointers in one chain -> suggest the legacy variant.
     - sub-64 / non-row-aligned access -> point at the padding helper.

6. DP KERNEL SELECTION.
   A dynamic-programming pass that selects the fastest kernel variant for each op
   and builds up the instruction blocks to accommodate it — choosing fused vs
   separate ops, PBI vs legacy, tile shapes — to minimize total runtime under the
   memory/route constraints from (1)-(5).

TARGET AUTHORING EXPERIENCE
---------------------------
The author writes ops at ORIGINAL dimensions; the tool auto-pads to the 64 quantum
and recycles DRAM. Composable op groups read like straight-line math:

    def init():
        # weights/config pulled + quant chosen + DRAM partitioned automatically
        ...

    sram_op_1 = (
        matmul()       # original dims; autopad + address assignment handled
        activation()
        matmul()
        norm()
        resid_add()
    )

    def forward():
        sram_op_1()

All the author passes is the original shapes. Padding, addressing, masks, quant,
mode selection, and kernel choice are the selector's job.

EXECUTION MODEL: compile in __init__, assemble in forward
---------------------------------------------------------
The class lifecycle is:
  __init__  : downloads the model, initializes everything (config, weights, DRAM
              partitions, masks/tables), then COMPILES the entire instruction stream.
              When __init__ returns, the forward pass already exists as compiled ops.
  prep()    : USER hook called by __init__. Declares weights (bf16 vs quantized
              helpers) and defines the op chains. Runs once at construction.
  exec()    : USER hook. Does NOT recompute anything — it just EXECUTES/assembles
              the already compiled op instructions in order (the forward pass).

Ops are emitted via `chain(...)` calls (see below). Each op is looked up in the
`UE_HLO` catalog (hlo.py) and lowered to a real engine kernel by `dispatch`; adjacent
ops may be merged by the `FUSIONS` DP. An unsupported op name is rejected with an
actionable message rather than emitting something that hangs the device.

CHAINING BACK-TO-BACK OPS (the first primitive we build)
--------------------------------------------------------
The lowest-level convenience is a `chain(...)` call that runs a sequence of ops
back-to-back, using the INPUT DIMS to size outputs and a running DRAM cursor to
place them — each op executes and then advances the cursor. Example:

    res = ue_s.chain(("mm", a, b, res), ("add", res, d, res))

Per-op flags ride in an optional TRAILING DICT (kwargs can't live in a tuple
literal), e.g. quant/activation/mode:

    res = ue_s.chain(("mm", a, b, res, {"act": "gelu", "pbi": True}), ("add", ...))

Each entry is a tuple `(op_name, *operands[, attrs_dict])` where the trailing
operand (before any dict) is the output slot. `chain` builds the op-sequence signature `("mm", "add")` and looks it
up in the FUSED_KERNELS hashtable:
  - if the whole sequence has a single fused kernel, it selects that ONE stub and
    runs it (one instruction block instead of several);
  - otherwise it falls back to running each op individually from SINGLE_OPS.
Either way, output Tensors are allocated from the current DRAM address (auto-padded
to 64) and the cursor is incremented, so the caller never computes an address.

STATUS
------
Skeleton only — this docstring is the spec. Implementation builds on the lib's
Tensor/Arena/Ops primitives (see selector.py) and is gated by the byte-identical
parity harness (tests/parity.md) before any in-tree adoption.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

# This file lives in <unified-engine>/kernel_select/. The engine modules
# (user_dma_core, nn_lib, quant_lib) live at the repo root = parent dir.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# PBI allow-sets per stage. Kept here (one place) so the <=3 advancing-pointer rule and
# the prefill/decode differences are a single audited policy, not per-model guesswork.
# flash stays legacy always.
PBI_ALLOW = {
    "prefill": {"matmul", "norm", "rope"},
    "decode":  {"matmul", "rope"},
}


def pad64(n: int) -> int:
    """Round up to the 64-element / 128-byte hardware quantum."""
    return ((n + 63) // 64) * 64


@dataclass
class Tensor:
    addr: int                      # DRAM byte address (assigned by the Arena)
    shape: tuple                   # logical shape, e.g. (S, H)
    dtype: str = "bf16"            # bf16 | q4_64 | fp4_e2m1
    bank: str = "A"                # 'A' activation (default) | 'B' weight
    name: str = ""                 # for debugging / parity dumps
    # filled by the arena; the padded-to-64 footprint actually resident in DRAM
    padded: tuple = field(default=None)
    # quantized weights carry their scale buffer here (None for bf16 / activations)
    scale: "Tensor" = field(default=None)

    BYTES = {"bf16": 2, "q4_64": 0.5, "fp4_e2m1": 0.5, "if4": 0.5,
             "int8": 1, "fp8": 1, "if8": 1}

    def __post_init__(self):
        if self.padded is None:
            self.padded = tuple(pad64(d) for d in self.shape)

    @property
    def numel_padded(self) -> int:
        n = 1
        for d in self.padded:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return int(self.numel_padded * self.BYTES[self.dtype])

    def __repr__(self):
        return (f"Tensor({self.name or '?'} {self.shape}->{self.padded} "
                f"{self.dtype}@0x{self.addr:X} bank={self.bank})")


# =============================================================================
# Arena — the ONE place DRAM addresses are decided. Three partitions mirror the
# engine's regions (params/tensor/program); we delegate the actual bump to the
# engine's own allocators so addresses match the hand-written models. The tensor
# partition adds *liveness reuse*: a scratch Tensor whose last use has passed
# returns its bytes to a free list, keeping the activation footprint flat
# regardless of layer count. Growing buffers (KV cache) take() permanently and
# are excluded from reuse.
# =============================================================================
class _Partition:
    def __init__(self, engine_alloc, name):
        self._alloc = engine_alloc      # bound UnifiedEngine.allocate_*_dram
        self.name = name
        self._free = []                 # [(addr, size)] reusable holes (tensor only)
        self._reuse_enabled = False

    def enable_reuse(self):
        self._reuse_enabled = True

    def take(self, size_bytes, label=None) -> int:
        if self._reuse_enabled:
            for i, (addr, sz) in enumerate(self._free):
                if sz >= size_bytes:
                    self._free.pop(i)
                    return addr
        return self._alloc(size_bytes, label=label)

    def give_back(self, addr, size_bytes):
        if self._reuse_enabled:
            self._free.append((addr, size_bytes))


class Arena:
    def __init__(self, ue):
        self.ue = ue
        self.params  = _Partition(ue.allocate_params_dram,  "params")
        self.tensor  = _Partition(ue.allocate_tensor_dram,  "tensor")
        self.program = _Partition(ue.allocate_program_dram, "program")
        self.tensor.enable_reuse()
        self._live = {}                 # id(Tensor) -> nbytes (for free())

    def alloc(self, shape, dtype="bf16", bank="A", name="", part="tensor") -> Tensor:
        t = Tensor(addr=0, shape=tuple(shape), dtype=dtype, bank=bank, name=name)
        t.addr = getattr(self, part).take(t.nbytes, label=name or None)
        if part == "tensor":
            self._live[id(t)] = t.nbytes
        return t

    def free(self, t: Tensor):
        """Mark a scratch tensor dead; its bytes become reusable."""
        sz = self._live.pop(id(t), None)
        if sz is not None:
            self.tensor.give_back(t.addr, sz)

    def alloc_weight(self, shape, dtype, name="") -> Tensor:
        return self.alloc(shape, dtype=dtype, bank="B", name=name, part="params")
from hlo import UE_HLO, dispatch     # noqa: E402  (public HLO catalog)


# =============================================================================
# Chain — run back-to-back ops, sizing outputs from input dims and placing them at
# a running DRAM cursor. Looks up the whole op sequence in FUSED_KERNELS first; if a
# fused kernel exists it runs that single stub, else falls back to per-op SINGLE_OPS.
#
# Entry format: (op_name, *operands) — the LAST operand is the output slot.
# Operands may be Tensor handles or string names (resolved in a per-chain table, so
# `("mm", a, b, "res")` then `("add", "res", d, "res")` works).
# =============================================================================
class Chain:
    def __init__(self, selector):
        self.s = selector                 # Ops-mixin host (emit methods)
        self.arena = selector.arena
        self._names: dict[str, Tensor] = {}

    # -- operand resolution -------------------------------------------------
    def _resolve(self, x):
        if isinstance(x, Tensor):
            return x
        if isinstance(x, str):
            return self._names.get(x)     # may be None until an op writes it
        raise TypeError(f"operand must be Tensor or str name, got {type(x)}")

    def _bind_out(self, slot, shape, dtype):
        """Allocate (or reuse) the output Tensor at the current DRAM cursor."""
        if isinstance(slot, Tensor):
            return slot
        t = self.arena.alloc(shape, dtype=dtype, name=str(slot))   # bumps the cursor
        if isinstance(slot, str):
            self._names[slot] = t
        return t

    @property
    def ue(self):
        return self.s.ue

    # dispatch() runs with the Chain as `host`, so the Stage-4 recorder reads these
    # off the Chain — proxy them to the selector where the real state lives.
    @property
    def _map_log(self):
        return self.s._map_log

    @property
    def _cur_program(self):
        return self.s._cur_program

    @property
    def _map_ctx(self):
        return self.s._map_ctx

    def _mode_reg(self, spec, attrs):
        """Decide PBI vs legacy for one op. Returns a GPR index for PBI, else None.
        Priority: explicit per-op {"M_reg": reg} > the program's loop_reg (named GPR,
        shared so a runtime-primed seq_len drives every op) > stage policy (fresh reg).
        Non-PBI kernels always get None."""
        if not spec.pbi:
            return None
        if "M_reg" in attrs:                       # explicit per-op register (None -> legacy)
            return None if attrs["M_reg"] is None else int(attrs["M_reg"])
        loop = getattr(self.s, "_loop_reg", None)  # shared loop register for the program
        if loop is not None:
            return int(loop)
        want = attrs.get("pbi")                    # fallback: stage policy, fresh reg
        if want is None:
            want = spec.name in getattr(self.s, "_pbi_allow", set())
        return self.s.ue.alloc_isa_reg() if want else None

    # -- entry parsing ------------------------------------------------------
    @staticmethod
    def _parse(entry):
        """(op_name, *operands[, attrs_dict]) -> (name, operands_tuple, attrs).
        A trailing dict is the per-op flags: pbi, act, causal, group, bias, ..."""
        name, rest = entry[0], list(entry[1:])
        attrs = rest.pop() if rest and isinstance(rest[-1], dict) else {}
        return name, tuple(rest), attrs

    # -- the call -----------------------------------------------------------
    def __call__(self, *entries):
        # 1. parse + precision-route each op (mm -> matmul/qmatmul) so the DP sees the
        #    REAL kernel names. Routing only probes the WEIGHT (2nd operand), which is a
        #    real Tensor available now — we do NOT resolve chained slots here (they
        #    aren't produced until emit time).
        routed = []
        for name, operands, attrs in (self._parse(e) for e in entries):
            probe = tuple(self._resolve(x) for x in operands[:-1])
            name, attrs = self._route(name, probe, attrs)
            routed.append((name, operands, attrs))
        names = [r[0] for r in routed]

        # 2. DP: cover the op sequence with the fewest kernels, preferring the longest
        #    available fusion at each step.
        plan = select_plan(names, FUSIONS)
        if self.s._map_log is not None:                # record the DP cover for the Stage-4 report
            self.s._plan_log.append({
                "program": self.s._cur_program, "ctx": self.s._map_ctx,
                "names": list(names), "segs": [(s, e, key) for (s, e, key) in plan],
            })

        # 3. emit IN ORDER. Resolve inputs at emit time so each op sees the buffers
        #    produced by the ops before it.
        out = None
        for start, end, key in plan:
            if key is not None:
                seg = [(routed[j][0], routed[j][1], routed[j][2]) for j in range(start, end)]
                out = FUSIONS[key](self, seg)
                if self.s._map_log is not None:
                    fins = [self._resolve(o) for o in routed[start][1][:-1]]
                    fprec = next((i.dtype for i in fins if getattr(i, "dtype", "bf16") != "bf16"), "bf16")
                    self.s._map_log.append({
                        "program": self.s._cur_program, "ctx": self.s._map_ctx,
                        "op": "+".join(key), "kernel": FUSION_LABEL.get(key, "(fused)"),
                        "src": "—", "pbi_capable": False, "mode": "fused", "prec": fprec,
                        "fused": list(key), "out": getattr(out, "name", ""),
                    })
            else:
                name, operands, attrs = routed[start]
                if name not in UE_HLO:
                    raise ValueError(f"unsupported op '{name}' (no kernel/fusion)")
                ins = tuple(self._resolve(x) for x in operands[:-1])
                out = dispatch(self, UE_HLO[name], ins, operands[-1], attrs)
        return out

    @staticmethod
    def _route(name, ins, attrs):
        """Op-name aliasing. 'mm' -> 'matmul'. The matmul kernel itself auto-detects a
        quantized weight (B.dtype != bf16) and adds is_B_quantized/data_type/SCALE from
        the weight's own .scale buffer — so the user always writes 'mm'/'matmul'
        regardless of precision; no separate quantized op name needed."""
        if name == "mm":
            return "matmul", attrs
        return name, attrs


# =============================================================================
# DP chain selection — cover the op sequence with the fewest kernels, preferring
# the longest registered fusion at each position. Classic interval DP:
#   dp[i] = min kernels to cover names[:i]; try a singleton, or any fusion of
#   length L (2..MAX) whose key matches names[i-L:i].
# Returns [(start, end, fusion_key_or_None)] in order.
# =============================================================================
FUSION_MAX_LEN = 4

def select_plan(names, fusions, max_len=FUSION_MAX_LEN):
    n = len(names)
    INF = float("inf")
    dp = [0] + [INF] * n
    choice = [None] * (n + 1)
    for i in range(1, n + 1):
        if dp[i - 1] + 1 < dp[i]:                       # singleton
            dp[i], choice[i] = dp[i - 1] + 1, (i - 1, None)
        for L in range(2, min(max_len, i) + 1):         # try fusions ending at i
            key = tuple(names[i - L:i])
            if key in fusions and dp[i - L] + 1 < dp[i]:
                dp[i], choice[i] = dp[i - L] + 1, (i - L, key)
    plan, i = [], n
    while i > 0:
        start, key = choice[i]
        plan.append((start, i, key))
        i = start
    return plan[::-1]


# -- fusion handlers: (chain, segment) -> out. segment = [(name, operands, attrs)].
#    A fusion exists only when the engine has ONE kernel for the whole segment.
def _fuse_ln_add(chain, seg):
    # layer_norm(x) then + residual  ->  layer_norm_core_dram_post_add
    (_, ln_ops, _), (_, add_ops, _) = seg
    x, gamma, beta = (chain._resolve(o) for o in ln_ops[:3])
    b, slot = chain._resolve(add_ops[1]), add_ops[-1]
    out = chain._bind_out(slot, x.shape, x.dtype)
    from nn_lib import layer_norm_core_dram_post_add   # nn_lib func, ue is first arg
    layer_norm_core_dram_post_add(
        chain.ue, M=x.shape[0], N=x.shape[1], A_DRAM_ADDR=x.addr, B_DRAM_ADDR=b.addr,
        OUTPUT_DRAM_ADDR=out.addr, GAMMA_DRAM_ADDR=gamma.addr, BETA_DRAM_ADDR=beta.addr)
    return out

def _fuse_rms_add(chain, seg):
    # rms_norm(x) then + residual  ->  rms_norm_core_dram_post_add  (gemma-style)
    (_, rn_ops, _), (_, add_ops, _) = seg
    x, gamma = chain._resolve(rn_ops[0]), chain._resolve(rn_ops[1])
    b, slot = chain._resolve(add_ops[1]), add_ops[-1]
    out = chain._bind_out(slot, x.shape, x.dtype)
    from nn_lib import rms_norm_core_dram_post_add
    rms_norm_core_dram_post_add(chain.ue, M=x.shape[0], N=x.shape[1],
        A_DRAM_ADDR=b.addr, B_DRAM_ADDR=x.addr, OUTPUT_DRAM_ADDR=out.addr,
        GAMMA_DRAM_ADDR=gamma.addr)
    return out

def _fuse_mm_add(chain, seg):
    # matmul then + bias  ->  matmat_mul_core(C_DRAM_ADDR=bias)
    (_, mm_ops, mm_attrs), (_, add_ops, _) = seg
    a, b = chain._resolve(mm_ops[0]), chain._resolve(mm_ops[1])
    d, slot = chain._resolve(add_ops[1]), add_ops[-1]
    out = chain._bind_out(slot, (a.shape[0], b.shape[1]), a.dtype)
    chain.ue.matmat_mul_core(M=pad64(a.shape[0]), K=pad64(a.shape[1]), N=pad64(b.shape[1]),
        A_DRAM_ADDR=a.addr, B_DRAM_ADDR=b.addr, OUTPUT_DRAM_ADDR=out.addr,
        C_DRAM_ADDR=d.addr, gelu_enable=(mm_attrs.get("act") == "gelu"),
        silu_enable=(mm_attrs.get("act") == "silu"), gpr_M_reg=None)
    return out

# keys are in ROUTED kernel-name space (mm is already -> matmul/qmatmul before DP).
# Only register a fusion when the engine has ONE kernel with the SAME semantics as the
# op sequence. matmul-then-add == matmat_mul_core(C_DRAM_ADDR=bias): correct, kept.
# NOTE: the *_post_add norm kernels compute norm(A+B) = ADD-then-NORM, which is a
# DIFFERENT sequence; they'd be keyed ("add","rms_norm"), not ("rms_norm","add"), and
# also emit two outputs. Left out until wired correctly (see _fuse_ln/_rms_add below).
FUSIONS = {
    ("matmul", "add"): _fuse_mm_add,
}

# Human-readable engine target per fusion, for the Stage-4 mapping report (the fused
# segment never goes through dispatch(), so it has no KernelSpec.fn to show otherwise).
FUSION_LABEL = {
    ("matmul", "add"): "matmat_mul_core (matmul + bias)",
}


# =============================================================================
# Reg — a named GPR handle. Subclasses int so it can be passed straight through as a
# register index anywhere the engine wants one (gpr_M_reg, gpr_bucket_idx, ...), while
# carrying a name for binding + debugging. Allocated once, reused across the program;
# the value is set at execute() time, so one captured program serves any seq_len.
# =============================================================================
class Reg(int):
    def __new__(cls, idx, name=""):
        r = super().__new__(cls, idx)
        r.name = name
        return r

    def __repr__(self):
        return f"Reg({int(self)}, '{self.name}')"


# =============================================================================
# Program capture — `with sel.program(name):` wraps the engine's capture lifecycle:
#   enter -> clear_inst_id + start_capture
#   exit  -> generate_instruction_halt + stop_capture + write stream to its own
#            program-DRAM slot (overflow -> actionable DRAM error). Records a Program.
# =============================================================================
@dataclass
class Program:
    name: str
    addr: int
    size: int


@dataclass
class GrowingBuffer:
    """A persistent, runtime-appendable region in TENSOR DRAM — e.g. a KV cache.

    Unlike normal activation tensors (which the Arena recycles via liveness reuse), a
    growing buffer is reserved at MAX capacity up front and never recycled, because it
    accumulates across execute() calls (one new row per decode token). Rows are written
    at a RUNTIME position via register-addressed DMA (see append_row)."""
    name: str
    base: int          # DRAM byte address of row 0
    row_bytes: int     # bytes per row (row_elems * dtype size)
    row_elems: int     # logical elements per row
    max_rows: int      # reserved capacity
    dtype: str = "bf16"

    @property
    def nbytes(self) -> int:
        return self.row_bytes * self.max_rows


class _ProgramCapture:
    def __init__(self, sel, name):
        self.sel, self.name = sel, name

    def __enter__(self):
        self.addr = self.sel.ue.get_program_dram_addr()   # this program's slot start
        self.sel._cur_program = self.name                 # tag map-recorder rows
        self.sel.ue.clear_inst_id()
        self.sel.ue.start_capture()
        return self

    def __exit__(self, *exc):
        self.sel._cur_program = None
        if exc[0] is not None:
            return False
        self.sel.ue.generate_instruction_halt()
        self.sel.ue.stop_capture()
        size = self.sel.ue.write_captured_instructions_to_dram(self.addr)
        cap = self.sel._program_capacity()
        used = (self.addr - self.sel.ue._program_dram_base) + size
        if cap is not None and used > cap:
            raise MemoryError(
                f"utilizing too much DRAM budget: program '{self.name}' overflows the "
                f"instruction region by {(used - cap) / 1024 / 1024:.0f} MB. Reduce shape "
                f"diversity or repartition away from params/tensors.")
        self.sel.ue.allocate_program_dram(size)           # advance the cursor
        self.sel.programs[self.name] = Program(self.name, self.addr, size)
        return False


# =============================================================================
# Ops — mixin providing per-op wrappers (add, mul, matmul, rms_norm, rope,
# flash_attn, etc.) that call the engine's core dispatchers. Each wrapper
# allocates the output Tensor, pads shapes to the 64 quantum, and picks
# PBI vs legacy via the mode policy.
# =============================================================================
class Ops:
    """Mixin holding self.ue (UnifiedEngine) and self.arena (Arena)."""

    # ---- elementwise -------------------------------------------------------
    def add(self, a: Tensor, b: Tensor, *, out=None) -> Tensor:
        from nn_lib import eltwise_add_core_dram
        out = out or self.arena.alloc(a.shape, a.dtype, name="add")
        size = a.numel_padded
        eltwise_add_core_dram(self.ue, size, a.addr, b.addr, out.addr)
        return out

    def mul(self, a: Tensor, b: Tensor, *, out=None) -> Tensor:
        from nn_lib import eltwise_mul_core_dram
        out = out or self.arena.alloc(a.shape, a.dtype, name="mul")
        eltwise_mul_core_dram(self.ue, a.numel_padded, a.addr, b.addr, out.addr)
        return out

    # ---- norms -------------------------------------------------------------
    def rms_norm(self, x: Tensor, gamma: Tensor, *, residual=None, out=None) -> Tensor:
        """rms_norm; if `residual` given uses the fused *_post_add variant."""
        out = out or self.arena.alloc(x.shape, x.dtype, name="rms_norm")
        M, N = x.shape[0], x.shape[1]
        if residual is not None:
            from nn_lib import rms_norm_core_dram_post_add
            rms_norm_core_dram_post_add(self.ue, M=M, N=N, A_DRAM_ADDR=residual.addr,
                                        B_DRAM_ADDR=x.addr, OUTPUT_DRAM_ADDR=out.addr,
                                        GAMMA_DRAM_ADDR=gamma.addr)
        else:
            self.ue.rms_norm_core_dram(M=M, N=N, A_DRAM_ADDR=x.addr,
                                       OUTPUT_DRAM_ADDR=out.addr, GAMMA_DRAM_ADDR=gamma.addr,
                                       gpr_M_reg=self._pbi_reg("norm"))
        return out

    def layer_norm(self, x: Tensor, gamma: Tensor, beta: Tensor, *, out=None) -> Tensor:
        out = out or self.arena.alloc(x.shape, x.dtype, name="layer_norm")
        self.ue.layer_norm_core_dram(M=x.shape[0], N=x.shape[1], A_DRAM_ADDR=x.addr,
                                     OUTPUT_DRAM_ADDR=out.addr, GAMMA_DRAM_ADDR=gamma.addr,
                                     BETA_DRAM_ADDR=beta.addr, gpr_M_reg=self._pbi_reg("norm"))
        return out

    # ---- matmul ------------------------------------------------------------
    def matmul(self, a: Tensor, w: Tensor, *, bias=None, act=None, out=None) -> Tensor:
        """a:(M,K) @ w:(K,N) -> (M,N). act in {None,'gelu','silu'}. Quant from w.dtype."""
        M, K = a.shape
        N = w.shape[1]
        out = out or self.arena.alloc((M, N), a.dtype, name="matmul")
        quant = w.dtype != "bf16"
        self.ue.matmat_mul_core(
            M=pad64(M), K=pad64(K), N=pad64(N),
            A_DRAM_ADDR=a.addr, B_DRAM_ADDR=w.addr, OUTPUT_DRAM_ADDR=out.addr,
            C_DRAM_ADDR=(bias.addr if bias is not None else None),
            is_B_quantized=quant,
            gelu_enable=(act == "gelu"), silu_enable=(act == "silu"),
            gpr_M_reg=self._pbi_reg("matmul"))
        return out

    # ---- rope --------------------------------------------------------------
    def rope(self, x: Tensor, cos: Tensor, sin: Tensor, *, out=None) -> Tensor:
        out = out or self.arena.alloc(x.shape, x.dtype, name="rope")
        self.ue.rope_hf_core_dram(M=x.shape[0], N=x.shape[1],
                                  input_dram_addr=x.addr, output_dram_addr=out.addr,
                                  cos_dram_addr=cos.addr, sin_dram_addr=sin.addr,
                                  gpr_M_reg=self._pbi_reg("rope"))
        return out

    # ---- attention (ALWAYS legacy path — see MODE policy) ------------------
    def flash_attn(self, q, k, v, *, causal, group, scratch, identity, mask, out=None) -> Tensor:
        from nn_lib import prefill_flash_attention_core
        out = out or self.arena.alloc(q.shape, q.dtype, name="attn")
        prefill_flash_attention_core(
            self.ue, head_dim=q.shape[-1], seq_len=q.shape[0],
            Q_DRAM_ADDR=q.addr, K_DRAM_ADDR=k.addr, V_DRAM_ADDR=v.addr,
            OUTPUT_DRAM_ADDR=out.addr, SCRATCH_DRAM_ADDR=scratch.addr,
            IDENTITY_DRAM_ADDR=identity.addr, BIAS_DRAM_ADDR=mask.addr,
            num_q_heads=group)
        return out

    # ---- mode policy -------------------------------------------------------
    def _pbi_reg(self, op_kind: str):
        """Return a GPR index to take the PBI path, or None for legacy.
        Driven by self._stage ('prefill'/'decode') and a per-stage allow-set so the
        advancing-pointer count stays <= 3. Set by set_stage()."""
        allow = getattr(self, "_pbi_allow", set())
        if op_kind in allow:
            return self.ue.alloc_isa_reg()
        return None


# =============================================================================
# Weight loading helpers — resolve precision per tensor name, quantize, store.
# =============================================================================
import fnmatch as _fnmatch

_PRECISION = {
    "bf16":     (True,  None),
    "q4_64":    (False, "int4"),
    "fp4_e2m1": (False, "fp4"),
    "if4":      (False, "if4"),
    "int8":     (False, "int8"),
    "fp8":      (False, "fp8"),
    "if8":      (False, "if8"),
}


def _resolve_precision(name: str, precision_map: dict) -> str:
    """Most-specific glob match wins; longest matching pattern by length, '*' last."""
    best, best_len = None, -1
    for pat, prec in precision_map.items():
        if _fnmatch.fnmatch(name, pat):
            score = len(pat) - pat.count("*")
            if score > best_len:
                best, best_len = prec, score
    if best is None:
        raise KeyError(f"no precision rule matches weight '{name}' (add a '*' default)")
    return best


def _store_bytes(ue, arena, raw: bytes, label: str) -> int:
    """Allocate params DRAM and DMA raw quantized bytes in. Returns the address."""
    from user_dma_core import DMA_DEVICE_H2C
    addr = arena.params.take(len(raw), label=label)
    ue.dma_write(DMA_DEVICE_H2C, addr, raw, len(raw))
    return addr


def digest_weights(ue, arena, state_dict, precision_map, *, on_store=None) -> dict:
    """Store each HF tensor at its resolved precision. Returns {name: Tensor}."""
    import os, sys, time
    from nn_lib import store_weight
    from quant_lib import quantize

    prof = os.environ.get("UE_PROF_PREP")          # set UE_PROF_PREP=1 for a quantize/DMA breakdown
    t_quant = t_dma = 0.0; n_q = 0
    handles = {}
    for name, tensor in state_dict.items():
        prec = _resolve_precision(name, precision_map)
        is_bf16, qarg = _PRECISION[prec]
        shape = tuple(tensor.shape)

        if is_bf16:
            t0 = time.perf_counter()
            addr = store_weight(ue, tensor)
            t_dma += time.perf_counter() - t0
            h = Tensor(addr=addr, shape=shape, dtype="bf16", bank="B", name=name)
        else:
            t0 = time.perf_counter()
            data_bytes, scale_bytes = quantize(qarg, tensor)
            t_quant += time.perf_counter() - t0; n_q += 1
            t0 = time.perf_counter()
            data_addr  = _store_bytes(ue, arena, data_bytes,  name + ".data")
            scale_addr = _store_bytes(ue, arena, scale_bytes, name + ".scale")
            t_dma += time.perf_counter() - t0
            h = Tensor(addr=data_addr, shape=shape, dtype=prec, bank="B", name=name)
            h.scale = Tensor(addr=scale_addr, shape=shape, dtype="bf16", bank="B",
                             name=name + ".scale")
        handles[name] = h
        if on_store:
            on_store(name, prec, h.nbytes)
    if prof:
        sys.__stdout__.write(f"\r  digest: quantize {t_quant:.1f}s ({n_q} tensors) · "
                             f"DMA-store {t_dma:.1f}s{' '*12}\n")
        sys.__stdout__.flush()
    return handles


# =============================================================================
# IRHarness — the class users subclass.
#
# Lifecycle:
#   __init__ : pull HF config, build engine + arena, then call prep() (user) and
#              compile the resulting op graph into one instruction stream.
#   prep()   : USER hook — download/declare weights (choosing bf16 vs quantized
#              helpers) and define the op chains. Runs ONCE at construction. A DRAM
#              overflow here raises the actionable "too much budget, quantize" error.
#   exec()   : USER hook — assemble + run the already-compiled ops (the forward pass).
#              Recomputes nothing; just executes the captured instruction stream.
# =============================================================================
class IRHarness(Ops):
    def __init__(self, hf_link: str = None, *, ue=None, verbose=False):
        self.hf_link = hf_link
        self.cfg = self._pull_hf_config(hf_link)
        self.ue = ue or self._make_engine()
        if not hasattr(self.ue, "bytes_per_element"):
            self.ue.bytes_per_element = 2          # bf16; some cores read this off the engine
        self.arena = Arena(self.ue)
        self._stage = "prefill"
        self._pbi_allow = set()                        # legacy mode by default
        # Op-mapping recorder (Stage-4 report). Off unless the subclass opts in via the
        # class flag `_capture_map`; then every chain()->UE_HLO resolution is logged as it
        # is emitted (ground truth, no re-compile). `_cur_program`/`_map_ctx` tag each row.
        self._cur_program = None
        self._map_ctx = None
        self._map_log = [] if getattr(type(self), "_capture_map", False) else None
        self._plan_log = [] if self._map_log is not None else None   # DP fusion-cover per chain()
        self.chain = Chain(self)                       # back-to-back op runner
        self.programs = {}                             # name -> Program (addr, size)
        self.growing = {}                              # name -> GrowingBuffer (KV caches)
        self._addr_tmp = self.ue.alloc_isa_reg()       # scratch GPR for runtime address math
        self.verbose = verbose                         # show engine's own prints?
        cls = type(self).__name__
        self._run_stage("Scanning & prep", self.prep)  # 1. weights + tables (live timer)
        self._run_stage("Compiling", self.build)        # 2. compile program(s) (live timer)
        total = sum(p.size for p in self.programs.values())
        progs = " ".join(f"{n}={p.size}B" for n, p in self.programs.items())
        print(f"  programs  {progs or 'none'}  ({total/1024/1024:.2f} MB)")
        if self.growing:
            gtot = sum(g.nbytes for g in self.growing.values())
            gcnt = len(self.growing)
            print(f"  buffers   {gcnt} growing  ({gtot/1024/1024:.1f} MB reserved)")

    # ---- weight declaration (used inside prep) -----------------------------
    def scan_weights(self, state_dict=None):
        """Load the HF state dict (no DMA yet) and expose the available weights so
        the user can pick which to lower. Returns {name: shape}; also sets
        self.weights (the shown dict). Everything defaults to bf16 until lowered."""
        self._state_dict = state_dict if state_dict is not None else self._load_state_dict()
        self._lowered = {}                       # name -> precision the user picked
        self.weights = {n: tuple(t.shape) for n, t in self._state_dict.items()}
        return self.weights

    def lower_precision(self, name: str, precision: str = "q4_64"):
        """Mark a weight (by its name from the scanned dict) to be quantized at
        load_weights() time. Unmarked weights stay bf16. To use a quantized matmul,
        the weight passed in must be one that was lowered here."""
        if not hasattr(self, "_lowered"):
            self._lowered = {}
        if name not in getattr(self, "weights", {}):
            raise KeyError(f"'{name}' is not in the scanned weights dict "
                           f"(call scan_weights() first; check the exact name)")
        self._lowered[name] = precision

    def set_precision(self, precision_map: dict):
        """Alternative bulk form: declare per-tensor precision by glob pattern,
        {glob: 'bf16'|'q4_64'|...}. Mutually exclusive with lower_precision()."""
        self._precision_map = precision_map

    def load_weights(self, state_dict=None):
        """Digest HF weights into params DRAM: weights named in lower_precision()
        get quantized, the rest stay bf16. Stores name->Tensor handles in self.W
        (quantized handles carry .scale). Raises an actionable DRAM error on params
        overflow."""
        sd = state_dict or getattr(self, "_state_dict", None) or self._load_state_dict()
        # build the precision map: explicit per-name lowering + bf16 default.
        pmap = getattr(self, "_precision_map", None)
        if pmap is None:
            pmap = {**getattr(self, "_lowered", {}), "*": "bf16"}
        self._precision_map = pmap
        used = [0]

        def _budget(name, prec, nbytes):
            used[0] += nbytes
            cap = self._params_capacity()
            if cap is not None and used[0] > cap:
                raise MemoryError(
                    f"utilizing too much DRAM budget: params region over by "
                    f"{(used[0]-cap)/1024/1024:.0f} MB at '{name}'. "
                    f"Use the quantized helpers (e.g. set '{name}' -> 'q4_64').")

        self.W = digest_weights(self.ue, self.arena, sd, self._precision_map,
                                on_store=_budget)
        return self.W

    def _params_capacity(self):
        """Bytes available in the params partition (tensor_base - params_base)."""
        try:
            return self.ue._tensor_dram_base - self.ue._params_dram_base
        except AttributeError:
            return None

    def _load_state_dict(self):
        raise NotImplementedError("fetch HF weights (snapshot_download + load) here")

    # ---- GPRs: named registers primed at execute() time --------------------
    def gpr(self, name: str) -> "Reg":
        """Allocate (or fetch) a named GPR. Use the handle as an op's loop register
        (loop_reg / {"M_reg": ...}) or bucket/pos reg. Its VALUE is set in execute()
        via bind=, so the compiled program is seq_len-agnostic."""
        if not hasattr(self, "_gprs"):
            self._gprs = {}
        if name not in self._gprs:
            self._gprs[name] = Reg(self.ue.alloc_isa_reg(), name)
        return self._gprs[name]

    def loop_reg(self, reg):
        """Set the default PBI loop register (gpr_M_reg) for ops in the current
        program. Ops auto-use it unless they pass their own {"M_reg": ...}."""
        self._loop_reg = reg

    # ---- Growing buffers (KV cache) + runtime-address ops ------------------
    def growing_buffer(self, name, max_rows, row_elems, dtype="bf16") -> "GrowingBuffer":
        """Reserve a persistent, runtime-appendable region in TENSOR DRAM (e.g. a KV
        cache). Reserves `max_rows` up front; the buffer fills one row at a time via
        append_row at a runtime position. Excluded from activation liveness reuse and
        tracked in self.growing so total reserved DRAM is visible in the build banner."""
        import torch
        row_bytes = int(row_elems * Tensor.BYTES[dtype])
        base = self.arena.tensor.take(row_bytes * max_rows, label=name)  # permanent, never freed
        # Zero-init the whole region. Decode attends a full 64-wide bucket and masks the
        # not-yet-written rows via the bias; if those rows held raw DRAM garbage (NaN/inf
        # bit patterns) the masked 0*V term becomes NaN. Reference zeroes its cache too.
        self.ue.dma_to_accelerator_memory(base, torch.zeros(max_rows, row_elems, dtype=torch.bfloat16))
        gb = GrowingBuffer(name, base, row_bytes, row_elems, max_rows, dtype)
        self.growing[name] = gb
        return gb

    def runtime_addr(self, base, pos_reg, stride_bytes):
        """REGISTER ARITHMETIC: compute, at runtime, addr = base + pos_reg*stride_bytes
        into the scratch address GPR, and return that GPR index. `pos_reg` is a GPR
        holding the current position (e.g. gpr_seq_len). Two engine instructions:
            tmp = pos * stride_bytes      (reg_mul_imm)
            tmp = tmp + base              (add_imm)
        """
        from user_dma_core import ue_35bit_addr_shifter
        tmp = self._addr_tmp
        self.ue.generate_instruction_reg_mul_imm(tmp, int(pos_reg), ue_35bit_addr_shifter(stride_bytes))
        self.ue.generate_instruction_add_imm(tmp, ue_35bit_addr_shifter(base), tmp)
        return tmp

    def advance(self, reg):
        """ON-DEVICE position advance: emit a single add-immediate-1 on `reg` so a
        decode program self-increments its position register at runtime. Call once at
        the END of a decode program (after lm_head) so re-running it walks the sequence
        without the host re-priming the register — mirrors gemma3_test's trailing
        generate_instruction_add_inc(gpr_seq_len). The register is device state and
        persists across program_execute calls."""
        self.ue.generate_instruction_add_inc(int(reg))

    def append_row(self, gbuf, src_addr, pos_reg):
        """REGISTER-ADDRESSED DMA: write one row (src_addr in DRAM) into `gbuf` at the
        runtime row `pos_reg`. Computes the destination = gbuf.base + pos*row_bytes via
        runtime_addr(), then a register-addressed DRAM copy. This is the KV-cache write:
        each decode step appends the new K (or V) row at the current position."""
        dst_reg = self.runtime_addr(gbuf.base, pos_reg, gbuf.row_bytes)
        self.ue.accelerator_memcpy(src_addr, 0, gbuf.row_bytes, gr_dst_addr=dst_reg)

    # ---- GQA row duplication (ported from gemma3 duplicate_gqa_rows_pbi) ----
    def gqa_duplicate(self, src_dram, dst_dram, *, head_dim, group_size, seq_reg, seq_len):
        """Expand a [seq, head_dim] K/V tensor to [seq, group_size, head_dim] by
        duplicating each token row `group_size` times, so GQA K/V matches the Q layout
        flash attention expects. Data movement (PBI memcpy loop), not a chainable op."""
        from user_dma_core import UE_VECTOR_SIZE
        ue = self.ue
        bpe = 2
        row_bytes = head_dim * bpe
        row_uram_words = row_bytes // (UE_VECTOR_SIZE * bpe)
        # stage source rows into SRAM, then duplicate out to DRAM
        ue.accelerator_memory_to_sram(src_dram, 0x10000, seq_len * head_dim)
        _, src_uram = ue.sram_address_to_uram_address(0x10000)
        ptr = ue.alloc_inst_ptr()
        ue.generate_instruction_pbi_init(
            dram_shared_addr=dst_dram, dma_length=row_bytes, output_size=0, uram_length=0,
            uram_a_start_addr=src_uram, uram_b_start_addr=src_uram, uram_wb_addr=0,
            uram_dst_addr=0, fmax_context_addr=0, inst_pointer_idx=ptr)
        ue.loop_start(loop_cnt=seq_len, gpr_loop_cnt=int(seq_reg))
        ue.loop_start(group_size)
        ue.sram_to_accelerator_memory(
            sram_address=0, accelerator_dram_address=row_bytes, element_size=head_dim,
            inst_pointer_idx=ptr, memcpy_length_bytes=0)
        ue.loop_end()
        ue.generate_instruction_pbi_inc(
            dram_shared_addr=0, dma_length=0, output_size=0, uram_length=0,
            uram_a_start_addr=row_uram_words, uram_b_start_addr=row_uram_words,
            uram_wb_addr=0, uram_dst_addr=0, fmax_context_addr=0, inst_pointer_idx=ptr)
        ue.loop_end()
        ue.release_inst_ptr(ptr)

    # ---- GQA attention block: q_norm/k_norm -> rope -> dup -> bucketed flash -
    def attention(self, q, k, v, *, q_norm=None, k_norm=None, cos, sin, group, head_dim,
                  scratch, identity, mask, attn_p, bucket_reg, qseq_reg,
                  k_cache=None, v_cache=None):
        """Full GQA attention from raw q/k/v projections. q is [seq, group*head_dim];
        k,v are [seq, head_dim]. Returns attn output [seq, group*head_dim]. Wires:
        optional per-head QK-norm (skipped if q_norm/k_norm is None, e.g. llama), rope
        (gqa on Q, plain on K), GQA K/V duplication, and the bucketized flash. Caller binds
        bucket_reg = aligned(seq*group)//64 and qseq_reg = seq*group at execute."""
        seq = q.shape[0]
        qs = seq * group
        # per-head QK-norm: view q as [seq*group, head_dim]. M = seq*group rows, so the
        # loop count is gpr_q_seq_len (qseq_reg). Skipped entirely when q_norm is None —
        # do NOT substitute a ones gamma (rms_norm would still divide by RMS).
        if q_norm is not None:
            q_hd = Tensor(addr=q.addr, shape=(qs, head_dim), dtype=q.dtype, name="q_hd")
            self.chain(("rms_norm", q_hd, q_norm, "qn_hd", {"M_reg": qseq_reg}))
            qn = Tensor(addr=self.chain._names["qn_hd"].addr, shape=(seq, group * head_dim),
                        dtype=q.dtype, name="qn")
        else:
            qn = Tensor(addr=q.addr, shape=(seq, group * head_dim), dtype=q.dtype, name="qn")
        kn = self.chain(("rms_norm", k, k_norm, "kn")) if k_norm is not None else k
        # rope: gqa on Q (grouped), plain on K
        qr = self.chain(("rope_gqa", qn, cos, sin, "qr", {"group": group}))
        # Populate the KV cache for decode: rope K straight into the cache (output slot =
        # cache base), exactly like the reference sets rope output_dram_addr to the cache.
        # Avoids a bulk DRAM->DRAM copy and guarantees the cache holds the roped K that
        # decode's per-position rope reproduces. V is the raw projection -> copy as-is.
        if k_cache is not None:
            kr_slot = Tensor(addr=k_cache.base, shape=(seq, head_dim), dtype=kn.dtype, name="kr")
            kr = self.chain(("rope", kn, cos, sin, kr_slot))
        else:
            kr = self.chain(("rope", kn, cos, sin, "kr"))
        if v_cache is not None:
            self.ue.accelerator_memcpy(v.addr, v_cache.base, seq * head_dim * Tensor.BYTES["bf16"])
        # duplicate K/V rows to [seq, group, head_dim] flat [qs, head_dim].
        # The flash bucket is 64-wide, so when qs isn't 64-aligned the rows [qs:bucket]
        # are read by the core but never written by gqa_duplicate. Zero them (whole
        # bucket) up front, or those raw-DRAM rows (NaN bit patterns) feed Q*K / softmax*V
        # and poison the result. Mirrors the reference's zero-padded K/V cache.
        import torch
        kd = self.arena.alloc((qs, head_dim), name="kd")
        vd = self.arena.alloc((qs, head_dim), name="vd")
        zpad = torch.zeros(pad64(qs), head_dim, dtype=torch.bfloat16)
        self.ue.dma_to_accelerator_memory(kd.addr, zpad)
        self.ue.dma_to_accelerator_memory(vd.addr, zpad)
        self.gqa_duplicate(kr.addr, kd.addr, head_dim=head_dim, group_size=group,
                           seq_reg=self.seq, seq_len=seq)
        self.gqa_duplicate(v.addr, vd.addr, head_dim=head_dim, group_size=group,
                           seq_reg=self.seq, seq_len=seq)
        # bucketized flash over q_seq = seq*group rows (Q viewed flat [qs, head_dim])
        qf = Tensor(addr=qr.addr, shape=(qs, head_dim), dtype=q.dtype, name="qf")
        a = self.chain(("flash_attn", qf, kd, vd, "attn",
                        {"group": group, "scratch": scratch, "identity": identity,
                         "mask": mask, "attn_p": attn_p, "bucket_reg": bucket_reg,
                         "num_buckets": max(1, (qs + 63) // 64)}))
        # view attn output back as [seq, group*head_dim]
        return Tensor(addr=a.addr, shape=(seq, group * head_dim), dtype=q.dtype, name="attn_out")

    # ---- DECODE attention: single token vs KV cache (gemma3_test-equivalent) -
    def decode_attention(self, q, k, v, *, q_norm=None, k_norm=None, rope_base, pos_reg, group,
                         head_dim, k_cache, v_cache, scratch, identity, mask, bucket_reg,
                         num_buckets=None):
        """Single-token decode attention. q:[1,group*head_dim], k,v:[1,head_dim].
        optional QK-norm (skipped if q_norm/k_norm None) -> rope_decode (K and per-head Q at
        runtime position) -> append K,V to the growing KV caches at pos -> grouped attention
        vs the cache. Mirrors the gemma3_test decode loop. Returns [1, group*head_dim]."""
        bpe = Tensor.BYTES["bf16"]
        # optional per-head QK-norm (group is small/compile-time -> legacy). Skip when None.
        if q_norm is not None:
            q_hd = Tensor(addr=q.addr, shape=(group, head_dim), dtype=q.dtype, name="dq_hd")
            self.chain(("rms_norm", q_hd, q_norm, "dqn", {"M_reg": None}))
            qn = self.chain._names["dqn"]
        else:
            qn = Tensor(addr=q.addr, shape=(group, head_dim), dtype=q.dtype, name="dqn")
        kn = self.chain(("rms_norm", k, k_norm, "dkn", {"M_reg": None})) if k_norm is not None else k
        # rope table row for this position, in a GPR. NOTE: `gr` is the shared _addr_tmp
        # register, and append_row() reuses _addr_tmp internally (runtime_addr) — so EVERY
        # op that reads `gr` (K rope + all Q-head ropes) MUST run BEFORE the appends, or the
        # appends clobber the rope-table address out from under the Q ropes (garbage Q).
        gr = self.runtime_addr(rope_base, pos_reg, 2 * head_dim * bpe)
        krope = self.chain(("rope_decode", kn, "dkr", {"gr_weight": gr}))   # [1, head_dim]
        # rope each Q head into a flash_q buffer [group, head_dim] (still using `gr`)
        flash_q = self.arena.alloc((group, head_dim), name="dfq")
        for g in range(group):
            qn_g = Tensor(addr=qn.addr + g * head_dim * bpe, shape=(1, head_dim), name=f"dqg{g}")
            out_g = Tensor(addr=flash_q.addr + g * head_dim * bpe, shape=(1, head_dim), name=f"dfqg{g}")
            self.chain(("rope_decode", qn_g, out_g, {"gr_weight": gr}))
        # appends LAST: they overwrite _addr_tmp, which is fine now that no rope needs `gr`.
        self.append_row(k_cache, krope.addr, pos_reg)                       # K cache[pos]
        self.append_row(v_cache, v.addr, pos_reg)                          # V cache[pos]
        kc = Tensor(addr=k_cache.base, shape=(k_cache.max_rows, head_dim), name="kc")
        vc = Tensor(addr=v_cache.base, shape=(v_cache.max_rows, head_dim), name="vc")
        # num_buckets is the COMPILE-TIME bucket count covering the full KV capacity
        # (ceil(max_rows/64)); the runtime bucket_reg selects how many to attend. The
        # PBI path ignores the static seq_len — bucket bodies cover K*64 each.
        from user_dma_core import UE_VECTOR_SIZE
        nb = num_buckets or max(1, (k_cache.max_rows + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE)
        a = self.chain(("group_attn", flash_q, kc, vc, "da",
                        {"group": group, "scratch": scratch, "identity": identity,
                         "mask": mask, "bucket_reg": bucket_reg, "num_buckets": nb}))
        return Tensor(addr=a.addr, shape=(1, group * head_dim), dtype=q.dtype, name="dattn")

    # ---- stage runner: live ticking timer while a phase runs ---------------
    def _run_stage(self, name, fn):
        """Run `fn` under _quiet() while a background thread prints a ticking elapsed
        timer to the real terminal, so long phases (prep/build) show progress instead
        of a frozen screen. Returns fn()'s result."""
        import threading
        import time
        real = sys.__stdout__                      # true terminal, bypasses _quiet redirect
        t0 = time.perf_counter()
        stop = threading.Event()

        def tick():
            while not stop.wait(0.1):
                real.write(f"\r{name} … {time.perf_counter() - t0:5.1f}s\033[K")
                real.flush()

        th = threading.Thread(target=tick, daemon=True)
        th.start()
        try:
            with self._quiet():
                return fn()
        finally:
            stop.set(); th.join()
            real.write(f"\r{name} → {time.perf_counter() - t0:.2f}s\033[K\n")
            real.flush()

    # ---- helper: silence the engine's own prints (unless verbose) ----------
    def _quiet(self):
        import contextlib
        import io
        if self.verbose:
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    # ---- helper: const tensor (alloc + DMA data in) ------------------------
    def const(self, data, name="const"):
        """Allocate a tensor slot and DMA bf16 data into it. For inputs/weights."""
        import torch
        t = self.arena.alloc(tuple(data.shape), name=name)
        self.ue.dma_to_accelerator_memory(t.addr, data.to(torch.bfloat16))
        return t

    # ---- user hooks --------------------------------------------------------
    def prep(self):
        """USER: declare/load weights, build rope tables + masks. Runs once."""
        raise NotImplementedError("subclass must implement prep()")

    def build(self):
        """USER: define the forward pass(es) inside `with self.program(name):`
        blocks. Each block auto-captures into its own program-DRAM slot. Runs once."""
        raise NotImplementedError("subclass must implement build()")

    # ---- program capture: clear/start_capture -> ops -> halt/stop/write -----
    def program(self, name):
        return _ProgramCapture(self, name)

    def execute(self, program="main", bind=None, banner=True):
        """Run a prebuilt program. `bind` primes named GPRs for this run, e.g.
            m.execute("prefill", bind={m.seq: S, m.qseq: S*G, m.bucket: align(S)//64})
        so one seq_len-agnostic program serves any prompt length. Registers are device
        state and persist into program_execute."""
        prog = self.programs[program]
        def _go():
            for reg, val in (bind or {}).items():
                self.ue.isa_add_set_core(int(reg), int(val))    # prime the GPR
            self.ue.program_execute(prog.addr)
        if banner:
            self._run_stage(program.capitalize(), _go)
        else:
            with self._quiet():
                _go()
        return self

    def _program_capacity(self):
        """Bytes available in the program partition (top of 4GB - program_base)."""
        try:
            top = self.ue._params_dram_base + (4 * 1024 ** 3)
            return top - self.ue._program_dram_base
        except AttributeError:
            return None

    def _pull_hf_config(self, hf_link):
        if hf_link is None:
            return {}
        raise NotImplementedError("fetch config.json from the HF link")

    def _make_engine(self):
        from user_dma_core import UnifiedEngine
        return UnifiedEngine()


__all__ = [
    "IRHarness", "Tensor", "pad64",
    "PBI_ALLOW", "Chain", "FUSIONS", "select_plan", "Reg", "GrowingBuffer", "Arena",
]
