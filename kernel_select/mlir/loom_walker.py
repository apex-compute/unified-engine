#!/usr/bin/env python3
"""loom_walker.py — Layer B execution: replay ue MLIR ops onto a real IRHarness.

Parses the (verified, window-raised) ue MLIR and replays each op as an
IRHarness.chain() call inside a ProgramPlan program, so PBI/Arena/dispatch are
inherited from the trusted path. The per-stage loop_reg drives PBI automatically:
PBI-capable ops (matmul/layer_norm/eltwise) collapse onto the stage register;
window/attention stay legacy.

This milestone = COMPILE-ONLY PBI GATE: build the program on the live engine,
capture it, and dump _map_log to verify each op's PBI mode. The attention cluster
(reshape/permute/select/flash) is DEFERRED (the non-PBI staging part) — its SSA
results are pre-allocated as placeholders so the matmul/eltwise chain stays wired.
"""
import os
import re
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir_harness import IRHarness, Tensor, pad64

# ops we emit as real kernels (rest are deferred / aliased)
_EMIT = {"layer_norm", "matmul", "eltwise", "window_partition", "window_reverse"}
_ALIAS = {"reshape", "select"}                       # pure views: forward the operand addr
_DEFER = {"permute", "flash_attn_pf"}                # attention staging — not emitted yet


# =============================================================================
# Minimal ue-MLIR parser
# =============================================================================
def _parse_shape(ts):
    m = re.search(r"tensor<([0-9x]+)x[a-z0-9]+>", ts)
    return [int(d) for d in m.group(1).split("x")] if m else None


def parse_mlir(text):
    """Return (args, ops). args=[(name,shape)]; ops=[dict(res,op,operands,attrs,out)]."""
    args, ops = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("func.func"):
            for nm, ty in re.findall(r"(%\w+):\s*(tensor<[^>]+>)", s):
                args.append((nm, _parse_shape(ty)))
            continue
        m = re.match(r"(%\w+) = ue\.(\w+) (.*?) : (.*)$", s)
        if not m:
            continue
        res, op, mid, types = m.groups()
        operands = re.findall(r"%\w+", mid.split("{")[0])
        attrs = {}
        ab = re.search(r"\{(.*)\}", mid)
        if ab:
            # value is either a [bracketed, list] or a comma-free scalar; match the
            # bracket form first so perm=[0, 2, 1, 3] isn't split on its commas.
            for k, v in re.findall(r"(\w+)\s*=\s*(\[[^\]]*\]|[^,]+)", ab.group(1)):
                attrs[k] = v.strip()
        out_shape = _parse_shape(types.split("->")[-1])
        ops.append(dict(res=res, op=op, operands=operands, attrs=attrs, out=out_shape))
    return args, ops


def _flat2d(shape):
    """ND -> (M, N): M = product of leading dims, N = last dim padded to the 64-ALU
    floor (the hardware requires N % 64 == 0). 1D (gamma/beta) -> (pad64(N),)."""
    if shape is None:
        return (1, 64)
    if len(shape) == 1:
        return (pad64(shape[0]),)
    return (int(math.prod(shape[:-1])), pad64(int(shape[-1])))


# =============================================================================
# The walker harness
# =============================================================================
class LoomWalker(IRHarness):
    _capture_map = True                         # enable _map_log ground truth

    def __init__(self, mlir_text, stage_reg, *, ue):
        self._args, self._ops = parse_mlir(mlir_text)
        self._stage_reg = stage_reg
        self.sym = {}                           # ssa name -> Tensor
        self.deferred = []                      # ops not yet emitted (attention staging)
        super().__init__(hf_link=None, ue=ue)

    def prep(self):
        # weights/inputs (func args) -> params; every SSA result -> a tensor slot.
        for nm, shp in self._args:
            self.sym[nm] = self.arena.alloc_weight(_flat2d(shp), "bf16", name=nm)
        for o in self._ops:
            self.sym[o["res"]] = self.arena.alloc(_flat2d(o["out"]), name=o["res"])

    def _attrs_for(self, o):
        op, a = o["op"], o["attrs"]
        ins = self.sym[o["operands"][0]]
        M, N = ins.shape if len(ins.shape) == 2 else (ins.shape[0], 1)
        if op == "matmul":
            act = a.get("act", '"none"').strip('"')
            return {"act": None if act == "none" else act}
        if op == "eltwise":
            return {"mode": a.get("mode", '"add"').strip('"')}
        if op in ("window_partition", "window_reverse"):
            side = int(round(math.sqrt(M)))     # token grid is square (H==W)
            return {"H": side, "W": side, "window_size": int(a.get("window_size", 7))}
        return {}

    def build(self):
        reg = self.gpr(self._stage_reg)
        with self.program("forward"):
            self.loop_reg(reg)                  # stage PBI register drives _mode_reg
            for o in self._ops:
                op = o["op"]
                if op in _ALIAS:                # view: result aliases the operand buffer
                    self.sym[o["res"]] = self.sym[o["operands"][0]]
                    continue
                if op in _DEFER:                # attention staging — placeholder stays
                    self.deferred.append(o)
                    continue
                if op in _EMIT:
                    ins = [self.sym[x] for x in o["operands"]]
                    out = self.sym[o["res"]]
                    attrs = self._attrs_for(o)
                    entry = (op, *ins, out) if not attrs else (op, *ins, out, attrs)
                    self.chain(entry)
                else:
                    raise ValueError(f"walker: unhandled ue op '{op}'")


_OPT = os.path.join(os.path.dirname(__file__), "llvm-project", "build", "bin", "standalone-opt")


def gen_stage_ops(dim, heads, res, ws=7):
    """Export a Swin block at this stage's dims -> ue MLIR -> raise -> parse."""
    import subprocess
    import torch
    import torch_to_ue
    m = torch_to_ue.SwinBlock(dim=dim, heads=heads, ws=ws).eval()
    mlir, _, _ = torch_to_ue.lower(m, (torch.randn(1, res, res, dim),), func_name=f"swin_d{dim}")
    p = subprocess.run([_OPT, "--pass-pipeline=builtin.module(ue-raise-windows)"],
                       input=mlir, capture_output=True, text=True)
    return parse_mlir(p.stdout if p.returncode == 0 else mlir)


def _attrs_for(sym, o):
    op, a = o["op"], o["attrs"]
    ins = sym[o["operands"][0]]
    M = ins.shape[0]
    if op == "matmul":
        act = a.get("act", '"none"').strip('"')
        return {"act": None if act == "none" else act}
    if op == "eltwise":
        return {"mode": a.get("mode", '"add"').strip('"')}
    if op in ("window_partition", "window_reverse"):
        side = int(round(math.sqrt(M)))
        return {"H": side, "W": side, "window_size": int(a.get("window_size", 7))}
    return {}


class FullSwinWalker(IRHarness):
    """Replay the WHOLE ProgramPlan into one `forward` program, switching the PBI
    loop register per stage. Each stage's block is replayed `depth` times."""
    _capture_map = True

    def __init__(self, plan, stage_data, *, ue):
        self._plan = plan
        self._stage_data = stage_data          # [(args, ops)] per stage
        super().__init__(hf_link=None, ue=ue)

    def prep(self):
        self._sw = []                          # per stage: (weights, input_arg, ops)
        for (args, ops) in self._stage_data:
            w, input_arg = {}, None
            for nm, shp in args:
                if shp and len(shp) == 4:      # the rank-4 activation = block input
                    input_arg = nm
                else:
                    w[nm] = self.arena.alloc_weight(_flat2d(shp), "bf16", name=nm)
            self._sw.append((w, input_arg, ops))

    def _emit_block(self, w, input_arg, ops, x):
        sym = dict(w)
        sym[input_arg] = x
        last = x
        for o in ops:
            op = o["op"]
            if op in _ALIAS:
                sym[o["res"]] = sym[o["operands"][0]]
            elif op in _DEFER:                 # attention staging: placeholder keeps chain wired
                sym[o["res"]] = self.arena.alloc(_flat2d(o["out"]), name=o["res"])
                self.deferred.append(op)
            elif op in _EMIT:
                ins = [sym[x] for x in o["operands"]]
                out = self.arena.alloc(_flat2d(o["out"]), name=o["res"])
                sym[o["res"]] = out
                attrs = _attrs_for(sym, o)
                entry = (op, *ins, out) if not attrs else (op, *ins, out, attrs)
                self.chain(entry)
            else:
                raise ValueError(f"walker: unhandled ue op '{op}'")
            last = sym[o["res"]]
        return last

    def build(self):
        self.deferred = []
        x = None
        with self.program("forward"):
            for si, stage in enumerate(self._plan.stages):
                self.loop_reg(self.gpr(stage.loop_reg))     # per-stage PBI register
                w, input_arg, ops = self._sw[si]
                x = self.arena.alloc((stage.tokens, pad64(stage.dim)), name=f"x_s{si}")
                for blk in range(stage.depth):
                    self._map_ctx = (si, blk)
                    x = self._emit_block(w, input_arg, ops, x)


def build_full_swin(plan, ue):
    """Generate per-stage IR from the plan, then compile the full forward program."""
    stage_data = [gen_stage_ops(s.dim, s.heads, s.resolution) for s in plan.stages]
    return FullSwinWalker(plan, stage_data, ue=ue)


def report_full(w):
    print(f"\n=== full-model _map_log — PBI mode per stage ===")
    print(f"{'stage':7s}{'op':18s}{'kernel':26s}{'mode':12s}out")
    print("-" * 76)
    by = {}
    for r in w._map_log:
        ctx = r["ctx"]
        by.setdefault(ctx[0] if ctx else -1, []).append(r)
    for si in sorted(by):
        regs = sorted({r["mode"] for r in by[si] if r["mode"].startswith("PBI")})
        print(f"-- stage {si}  ({len(by[si])} ops, PBI regs: {regs}) --")
        for r in by[si][:6]:
            print(f"{'':7s}{r['op']:18s}{r['kernel']:26s}{r['mode']:12s}{r['out']}")
    tot = len(w._map_log)
    pbi = sum(1 for r in w._map_log if r["mode"].startswith("PBI"))
    print(f"\ntotal emitted ops={tot}  PBI={pbi}  legacy={tot-pbi}  "
          f"deferred={len(w.deferred)}")


def report(walker):
    print(f"\n=== _map_log (compile-only PBI gate) — program 'forward' ===")
    print(f"{'op':18s}{'kernel':26s}{'mode':12s}{'pbi?':6s}out")
    print("-" * 78)
    for r in walker._map_log:
        print(f"{r['op']:18s}{r['kernel']:26s}{r['mode']:12s}"
              f"{str(r['pbi_capable']):6s}{r['out']}")
    if walker.deferred:
        print(f"\ndeferred (attention staging, not emitted): "
              f"{[o['op'] for o in walker.deferred]}")


if __name__ == "__main__":
    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        import loom_plan
        plan = loom_plan.plan_for("microsoft/swin-tiny-patch4-window7-224")
        w = build_full_swin(plan, ue)
        report_full(w)
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
            os.path.dirname(__file__), "swin_block.raised.mlir")
        with open(path) as f:
            text = f.read()
        w = LoomWalker(text, stage_reg="tok_s0", ue=ue)
        report(w)
