#!/usr/bin/env python3
"""torch_to_ue.py — emit `ue` dialect MLIR from a torch.export FX graph.

This is the FRONTEND (ATen -> ue lowering), v1. It walks the exported ATen graph
node-by-node and emits ue ops, in graph order. The mapping is a reusable rule
table keyed by ATen target — new encoders reuse the same rules.

v1 scope:
  1:1   aten.layer_norm                -> ue.layer_norm
        aten.linear                    -> ue.matmul   (+ fold following gelu -> act="gelu")
        aten.scaled_dot_product_attention -> ue.flash_attn_pf
        aten.add                       -> ue.eltwise mode="add"
  fold  view/reshape/permute/transpose/expand/contiguous/_unsafe_view -> value-forward
        (pure shape bookkeeping; windowing as explicit ue.window_partition/reverse
         is the NEXT pattern to lift — flagged, not yet emitted)

Run:  python torch_to_ue.py            # exports a Swin block, prints ue MLIR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- shape ops that carry no compute: forward the underlying value ----------
_SHAPE_OPS = {
    "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
    "aten.permute.default", "aten.transpose.int", "aten.expand.default",
    "aten.contiguous.default", "aten.flatten.using_ints", "aten.clone.default",
}


def _branch_nodes(node, stop):
    """DFS-collect the fx nodes feeding `node` (a bias/mask branch), stopping at
    placeholders/params or any node in `stop`. Used to consume the relpos/mask
    sub-graph that feeds the attention score add."""
    out, stack = set(), [node]
    while stack:
        x = stack.pop()
        if not isinstance(x, torch.fx.Node) or x in out or x in stop or x.op == "placeholder":
            continue
        out.add(x)
        for a in x.args:
            if isinstance(a, torch.fx.Node):
                stack.append(a)
            elif isinstance(a, (list, tuple)):
                stack.extend(y for y in a if isinstance(y, torch.fx.Node))
    return out


def _recognize_attention(gm):
    """THE FLASH RECOGNIZER. Find eager attention spans regardless of how torch
    decomposed them:  matmul(Q,Kᵀ) -> [div] -> [+relpos] -> [+mask/masked_fill]
    -> softmax -> [dropout] -> matmul(·,V).  Anchored on softmax. Returns
    (consumed_nodes, anchor2desc) where anchor = the PV matmul (emit one ue.flash
    there) and desc carries q/k/v + relpos/mask presence. This is what funnels BOTH
    eager and sdpa attention into the unified flash op -> selector -> kernel."""
    nodes = list(gm.graph.nodes)
    consumers = {}
    for n in nodes:
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                consumers.setdefault(a, []).append(n)
            elif isinstance(a, (list, tuple)):
                for y in a:
                    if isinstance(y, torch.fx.Node):
                        consumers.setdefault(y, []).append(n)

    def _untranspose(x):
        return x.args[0] if (isinstance(x, torch.fx.Node) and "transpose" in str(x.target)) else x

    consumed, anchor = set(), {}
    for s in nodes:
        if str(s.target) != "aten.softmax.int":
            continue
        # back-chain scores -> QK matmul, gathering relpos/mask bias branches
        chain, relpos, mask, cur = [], None, None, s.args[0]
        while isinstance(cur, torch.fx.Node) and str(cur.target) != "aten.matmul.default":
            chain.append(cur); t = str(cur.target)
            if t == "aten.add.Tensor":
                br = _branch_nodes(cur.args[1], set())
                if any("index" in str(x.target) for x in br):
                    relpos = cur.args[1]
                else:
                    mask = cur.args[1]
                consumed |= br
                cur = cur.args[0]
            elif t in ("aten.div.Tensor", "aten.mul.Tensor"):
                cur = cur.args[0]
            elif t == "aten.masked_fill.Scalar":
                mask = cur; cur = cur.args[0]
            else:
                cur = cur.args[0] if cur.args else None
        if not (isinstance(cur, torch.fx.Node) and str(cur.target) == "aten.matmul.default"):
            continue
        qk = cur
        Q, K = qk.args[0], _untranspose(qk.args[1])
        # forward softmax -> (dropout) -> PV matmul
        nxt = (consumers.get(s) or [None])[0]
        drop = None
        if nxt is not None and "dropout" in str(nxt.target):
            drop = nxt; nxt = (consumers.get(nxt) or [None])[0]
        if not (nxt is not None and str(nxt.target) == "aten.matmul.default"):
            continue
        pv = nxt
        V = _untranspose(pv.args[1])
        consumed |= set(chain) | {s, qk}
        if drop is not None:
            consumed.add(drop)
        anchor[pv] = dict(q=Q, k=K, v=V, relpos_node=relpos, mask_node=mask,
                          has_relpos=relpos is not None, has_mask=mask is not None)
    return consumed, anchor


def _recognize_patch_merge(gm):
    """Recognize Swin patch-merging: cat([x[0::2,0::2], x[1::2,0::2], x[0::2,1::2],
    x[1::2,1::2]], -1). Anchor on the 4-way cat, trace its operands back through
    slice/view to the common source, collapse to one ue.patch_merge. Returns
    (consumed, anchor: cat_node -> {src})."""
    consumed, anchor = set(), {}
    for n in gm.graph.nodes:
        if str(n.target) != "aten.cat.default":
            continue
        operands = n.args[0]
        if not (isinstance(operands, (list, tuple)) and len(operands) == 4):
            continue
        seen, stack, inputs, span = set(), list(operands), set(), set()
        while stack:
            x = stack.pop()
            if not isinstance(x, torch.fx.Node) or x in seen:
                continue
            seen.add(x); t = str(x.target)
            if "slice" in t or "view" in t or "reshape" in t or "clone" in t:
                span.add(x)
                stack += [a for a in x.args if isinstance(a, torch.fx.Node)]
            else:
                inputs.add(x)
        if len(inputs) == 1:
            anchor[n] = dict(src=next(iter(inputs)))
            consumed |= span
    return consumed, anchor


def _fold_constants(gm, node2val, user_inputs):
    """Evaluate every input-INDEPENDENT call_function on the HOST (the attention mask
    img-construction, relpos gathers, etc.). Returns (env, dep): env maps each foldable
    node -> its concrete tensor; dep is the set of input-dependent nodes. This is the
    constant-folding pass that turns the mask machinery into precomputed constants."""
    dep = set(user_inputs)
    for n in gm.graph.nodes:
        if n.op == "call_function" and any(x in dep for x in n.all_input_nodes):
            dep.add(n)
    env = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            if n in node2val and node2val[n] is not None:
                env[n] = node2val[n]
        elif n.op == "call_function" and n not in dep:
            if any(x not in env for x in n.all_input_nodes):  # an input didn't fold
                continue
            def resolve(a):
                if isinstance(a, torch.fx.Node):
                    return env.get(a)
                if isinstance(a, (list, tuple)):
                    return type(a)(resolve(y) for y in a)
                return a
            try:
                args = tuple(resolve(a) for a in n.args)
                kwargs = {k: resolve(v) for k, v in n.kwargs.items()}
                env[n] = n.target(*args, **kwargs)
            except Exception:
                pass
    return env, dep


class UnsupportedOp(Exception):
    """Raised when the model uses ATen ops with no ue-HLO kernel mapping.
    Carries the FULL gap list (every missing op + shape) so it reads as a
    coverage to-do, not a single fail point."""
    def __init__(self, gaps):
        self.gaps = gaps                    # [(target, shape)]
        names = sorted({g[0] for g in gaps})
        super().__init__(
            f"{len(gaps)} unsupported ATen op(s) — no ue-HLO kernel maps these:\n  "
            + "\n  ".join(names)
            + "\nAdd a rule in torch_to_ue.Emitter.handle (+ a KernelSpec in hlo.py) "
              "for each, or pass strict=False to emit a poisoned ue.unsupported marker.")


def _bf16_ty(shape):
    dims = "x".join(str(int(d)) for d in shape)
    return f"tensor<{dims}xbf16>"


def _shape_of(node):
    v = node.meta.get("val")
    if v is None or not hasattr(v, "shape"):
        return None
    return tuple(int(d) for d in v.shape)


class Emitter:
    def __init__(self, patch_spec=None):
        self.n = 0
        self.val = {}        # fx node -> ue ssa name (str)
        self.shape = {}      # fx node -> shape tuple
        self.lines = []
        self.folded = []     # report: ops folded as shape-only
        self.unsupported = []  # [(target, shape)] — ops with no ue-HLO kernel
        self.pending_act = {}  # ue ssa name -> act, for gelu fold
        # Layer-A declared patch embedding (in_ch/out_dim/patch). A conv is treated
        # as patch-embed ONLY if it matches this — role-based, not stride-guessed.
        self.patch_spec = patch_spec
        self._pe_alias = set()  # fx nodes that are the conv->flatten->transpose tail
        # binding: result-ssa -> real weight tensors for that op (weight/bias/gamma/...).
        # The IR gives op order+shapes; this gives the DATA the executor loads per op.
        self.node2val = {}      # fx placeholder node -> real tensor (set by lower)
        self.op_weights = {}    # result-ssa -> {role: tensor}
        self.attn_consumed = set()   # fx nodes subsumed by a recognized flash span
        self.attn_anchor = {}        # PV matmul node -> attention descriptor
        self.const_consumed = set()  # input-independent nodes folded away on host
        self.merge_consumed = set()  # slice/view nodes subsumed by a patch_merge
        self.merge_anchor = {}       # cat node -> {src} (patch-merge gather)
        # THE PRECOMPUTABLES SECTION: one spot for every host-computed constant that
        # gets DMA'd like a weight — attention masks, relpos bias, scales, identity.
        self.precomputed = {}        # name -> tensor
        self.flash_bias = {}         # flash anchor node -> {relpos: name, mask: name}

    def fresh(self):
        s = f"%{self.n}"; self.n += 1; return s

    def ssa(self, node):
        """ue ssa name for an fx node (placeholder or computed)."""
        return self.val.get(node, "%UNK")

    def emit(self, node, op, operand_nodes, attrs="", out_shape=None):
        dst = self.fresh()
        ins = ", ".join(self.ssa(o) for o in operand_nodes)
        in_tys = ", ".join(_bf16_ty(self.shape[o]) for o in operand_nodes)
        out_shape = out_shape if out_shape is not None else _shape_of(node)
        self.shape[node] = out_shape
        self.val[node] = dst
        a = f" {{{attrs}}}" if attrs else ""
        self.lines.append(
            f"    {dst} = ue.{op} {ins}{a} : ({in_tys}) -> {_bf16_ty(out_shape)}")
        return dst

    def _mark_unsupported(self, node, t, args):
        """Record an unsupported op + emit a poisoned `ue.unsupported` marker so the
        scan continues and the FULL gap list is collected. lower() raises at the end."""
        self.folded.append(f"UNHANDLED:{t}")
        self.unsupported.append((t, _shape_of(node)))
        self.shape[node] = _shape_of(node) or (self.shape.get(args[0]) if args else None)
        dst = self.fresh()
        self.val[node] = dst
        in_tys = ", ".join(_bf16_ty(self.shape[o]) for o in args
                           if self.shape.get(o) is not None)
        out_ty = _bf16_ty(self.shape[node]) if self.shape[node] else "tensor<1xbf16>"
        ins = ", ".join(self.ssa(o) for o in args)
        self.lines.append(
            f'    {dst} = ue.unsupported {ins} {{aten = "{t}"}} : ({in_tys}) -> {out_ty}')
        return dst

    # ---- per-ATen-op rules ---------------------------------------------------
    @staticmethod
    def _conv_params(node, t):
        """Extract (stride, pad, dilation, groups) from a conv node, honoring the
        different arg layouts + omitted trailing defaults of the two ATen forms:
          aten.conv2d.default:      (in, w, bias?, stride=1, pad=0, dil=1, groups=1)
          aten.convolution.default: (in, w, bias, stride, pad, dil, transposed, opad, groups)"""
        a = node.args
        def pair(v, d):
            if v is None: return [d, d]
            return list(v) if isinstance(v, (list, tuple)) else [int(v), int(v)]
        if t == "aten.convolution.default":
            return (pair(a[3], 1), pair(a[4], 0), pair(a[5], 1), int(a[8]))
        # conv2d.default — trailing args may be absent (at their defaults)
        g = a[6] if len(a) > 6 else 1
        return (pair(a[3] if len(a) > 3 else None, 1),
                pair(a[4] if len(a) > 4 else None, 0),
                pair(a[5] if len(a) > 5 else None, 1), int(g))

    def _is_patch_embed(self, node):
        """Role-based: a conv IS the patch embedding ONLY if it matches the
        Layer-A declared patch_spec (in_ch/out_dim/patch) AND is a true
        non-overlapping patchify (stride==kernel, no pad/dilation, groups=1).
        Every other conv -> a real conv -> unsupported (fail closed)."""
        ps = self.patch_spec
        if ps is None:
            return False
        w = node.args[1]
        wshape = self.shape.get(w) or _shape_of(w)
        if not wshape or len(wshape) != 4:
            return False
        out_ch, in_ch, kh, kw = wshape
        stride, pad, dil, groups = self._conv_params(node, str(node.target))
        p = int(ps["patch"])
        return (in_ch == int(ps["in_ch"]) and out_ch == int(ps["out_dim"])
                and [kh, kw] == [p, p] and stride == [p, p]
                and pad == [0, 0] and dil == [1, 1] and groups == 1)

    def handle(self, node, gm):
        t = str(node.target)
        args = [a for a in node.args if isinstance(a, torch.fx.Node)]

        # --- FLASH RECOGNIZER: collapse a recognized eager attention span ---
        if node in self.attn_anchor:
            d = self.attn_anchor[node]
            qsh = self.shape.get(d["q"]) or _shape_of(d["q"])
            heads = qsh[1] if qsh and len(qsh) >= 3 else 1
            extra = ""
            if d["has_relpos"]:
                extra += ", relpos = true"
            if d["has_mask"]:
                extra += ", mask = true"
            return self.emit(node, "flash_attn_pf", [d["q"], d["k"], d["v"]],
                             attrs=f"causal = false, heads = {heads} : i64{extra}")
        if node in self.attn_consumed or node in self.const_consumed or node in self.merge_consumed:
            self.shape[node] = _shape_of(node)
            self.val[node] = "%consumed"        # folded constant / span-internal
            return self.val[node]

        # --- patch-merge gather (collapsed slice/cat idiom) ---
        if node in self.merge_anchor:
            src = self.merge_anchor[node]["src"]
            csh = _shape_of(node)               # cat out [B, H/2, W/2, 4C]
            import math as _m
            out_shape = (int(_m.prod(csh[:-1])), csh[-1]) if csh else None
            return self.emit(node, "patch_merge", [src], out_shape=out_shape)

        if t == "aten.dropout.default":          # inference: identity
            src = node.args[0]
            self.val[node] = self.ssa(src)
            self.shape[node] = self.shape.get(src) or _shape_of(node)
            return self.val[node]

        if t == "aten.roll.default":             # shifted-window cyclic shift
            inp = node.args[0]
            shifts = node.args[1] if len(node.args) > 1 else []
            dims = node.args[2] if len(node.args) > 2 else []
            sh = ", ".join(str(int(s)) for s in (shifts if isinstance(shifts, (list, tuple)) else [shifts]))
            dm = ", ".join(str(int(d)) for d in (dims if isinstance(dims, (list, tuple)) else [dims]))
            return self.emit(node, "roll", [inp], attrs=f"shifts = [{sh}], dims = [{dm}]")

        if t in ("aten.pad.default", "aten.constant_pad_nd.default"):
            inp, pads = node.args[0], node.args[1]
            if all(int(p) == 0 for p in pads):   # no-op pad (grid already aligned) -> identity
                self.val[node] = self.ssa(inp)
                self.shape[node] = self.shape.get(inp) or _shape_of(node)
                return self.val[node]
            pj = ", ".join(str(int(p)) for p in pads)
            return self.emit(node, "pad", [inp], attrs=f"pads = [{pj}]")

        if t == "aten.adaptive_avg_pool1d.default":   # final mean pool over sequence
            return self.emit(node, "mean", [node.args[0]])

        # patch-embed tail: HF does `conv(...).flatten(2).transpose(1,2)`. patching
        # ALREADY outputs token-major (tokens, embed), so the trailing flatten/
        # transpose/reshape are no-ops here — forward the value, do NOT emit a real
        # permute (which would double-transpose and corrupt the layout).
        if (args and args[0] in self._pe_alias and
                t in (_SHAPE_OPS | {"aten.transpose.int", "aten.permute.default",
                                    "aten.flatten.using_ints"})):
            # alias the patching value AND its (token-major) type — the logical FX
            # shapes ([1,embed,tokens] etc.) must NOT leak, or uses of this SSA value
            # disagree on type. patching already produced the final (tokens, embed).
            self.val[node] = self.ssa(args[0])
            self.shape[node] = self.shape.get(args[0])
            self._pe_alias.add(node)
            self.folded.append(f"PE_FOLD:{t}")
            return self.val[node]

        if t in ("aten.convolution.default", "aten.conv2d.default"):
            if self._is_patch_embed(node):
                inp, w = node.args[0], node.args[1]
                ish = self.shape.get(inp) or _shape_of(inp)   # [B, C, H, W]
                p = int(self.patch_spec["patch"])
                embed = int(self.patch_spec["out_dim"])
                tokens = (ish[2] // p) * (ish[3] // p)
                dst = self.emit(node, "patching", [inp, w],
                                attrs=f"patch_size = {p} : i64",
                                out_shape=(tokens, embed))
                self._pe_alias.add(node)           # absorb the flatten/transpose tail
                return dst
            # a REAL conv (overlap/pad/dilation/groups, or not the declared
            # patch-embed) — no kernel for it -> fail closed, named in the gap list.
            return self._mark_unsupported(node, t, args)

        if t == "aten.layer_norm.default":
            # args: (input, normalized_shape, weight, bias, eps)
            inp, w, b = node.args[0], node.args[2], node.args[3]
            dst = self.emit(node, "layer_norm", [inp, w, b])
            self.op_weights[dst] = {"gamma": self.node2val.get(w),
                                    "beta": self.node2val.get(b)}
            return dst

        if t == "aten.linear.default":
            inp, w = node.args[0], node.args[1]
            b = node.args[2] if len(node.args) > 2 else None
            dst = self.emit(node, "matmul", [inp, w])
            self.op_weights[dst] = {"weight": self.node2val.get(w),
                                    "bias": self.node2val.get(b) if b is not None else None}
            return dst

        if t == "aten.gelu.default":
            # fold into the producing matmul: rewrite its line to act="gelu"
            src = node.args[0]
            src_ssa = self.ssa(src)
            for i, ln in enumerate(self.lines):
                if ln.strip().startswith(src_ssa + " = ue.matmul"):
                    self.lines[i] = ln.replace(" : (", ' {act = "gelu"} : (', 1)
                    break
            # forward value: downstream sees the (now gelu'd) matmul result
            self.val[node] = src_ssa
            self.shape[node] = self.shape.get(src)
            return src_ssa

        if t == "aten.scaled_dot_product_attention.default":
            q, k, v = node.args[0], node.args[1], node.args[2]   # carry all three
            heads = self.shape[q][1] if self.shape.get(q) and len(self.shape[q]) >= 3 else 1
            return self.emit(node, "flash_attn_pf", [q, k, v],
                             attrs=f"causal = false, heads = {heads} : i64")

        if t in ("aten.add.Tensor", "aten.add.default"):
            if len(args) == 2:
                return self.emit(node, "eltwise", args, attrs='mode = "add"')

        if t == "aten.permute.default":
            inp, perm = node.args[0], node.args[1]
            pj = ", ".join(str(int(p)) for p in perm)
            return self.emit(node, "permute", [inp], attrs=f"perm = [{pj}]")

        if t == "aten.transpose.int":
            inp, d0, d1 = node.args[0], int(node.args[1]), int(node.args[2])
            rank = len(self.shape[inp])
            d0 %= rank; d1 %= rank
            perm = list(range(rank)); perm[d0], perm[d1] = perm[d1], perm[d0]
            pj = ", ".join(str(p) for p in perm)
            return self.emit(node, "permute", [inp], attrs=f"perm = [{pj}]")

        if t == "aten.select.int":
            inp, dim, idx = node.args[0], int(node.args[1]), int(node.args[2])
            return self.emit(node, "select", [inp],
                             attrs=f"dim = {dim} : i64, index = {idx} : i64")

        if t in _SHAPE_OPS:   # view/reshape/_unsafe_view/flatten/expand/contiguous/clone
            return self.emit(node, "reshape", [node.args[0]])

        # unknown op: NO silent identity. Record the gap + emit a poisoned marker
        # so the whole graph is scanned; lower() raises UnsupportedOp at the end.
        return self._mark_unsupported(node, t, args)


def lower(model, example_inputs, func_name="swin_block", strict=True, patch_spec=None):
    """patch_spec: Layer-A declared patch embedding {in_ch, out_dim, patch}. When
    given, a matching conv lowers to ue.patching; any other conv fails closed."""
    ep = torch.export.export(model, example_inputs)
    gm = ep.graph_module
    em = Emitter(patch_spec=patch_spec)

    # placeholders -> block args
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    arg_decls = []
    for n in placeholders:
        s = em.fresh()
        em.val[n] = s
        em.shape[n] = _shape_of(n)
        arg_decls.append(f"{s}: {_bf16_ty(em.shape[n])}")

    # ordered real values per placeholder (param/buffer->state_dict, input->example).
    # Build the binding FIRST so handle() can record real weights per op.
    from torch.export.graph_signature import InputKind
    sd = ep.state_dict
    argvals, ui = [], list(example_inputs)
    for spec in ep.graph_signature.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            argvals.append(sd[spec.target])
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            argvals.append(ep.constants[spec.target])
        else:                                   # USER_INPUT
            argvals.append(ui.pop(0))
    em.node2val = dict(zip(placeholders, argvals))
    em.attn_consumed, em.attn_anchor = _recognize_attention(gm)
    em.merge_consumed, em.merge_anchor = _recognize_patch_merge(gm)

    # constant-fold: evaluate input-independent nodes on host, then consume the dead
    # constant subgraph (mask/relpos construction) whose every consumer is itself a
    # constant or subsumed by a flash span. Their values go to the precomputables spot.
    user_inputs = [placeholders[i] for i, spec in enumerate(ep.graph_signature.input_specs)
                   if spec.kind == InputKind.USER_INPUT]
    env, dep = _fold_constants(gm, em.node2val, user_inputs)
    const_nodes = {n for n in gm.graph.nodes if n.op == "call_function" and n not in dep}
    consumers = {}
    for n in gm.graph.nodes:
        for a in n.all_input_nodes:
            consumers.setdefault(a, []).append(n)
    dead = const_nodes | em.attn_consumed
    em.const_consumed = {n for n in const_nodes
                         if n in env and all(c in dead for c in consumers.get(n, []))}
    # stash each flash op's relpos/mask as precomputed bias constants (the ONE spot)
    for i, (anchor, d) in enumerate(em.attn_anchor.items()):
        for role in ("relpos", "mask"):
            nd = d.get(f"{role}_node")
            if nd is not None and nd in env:
                nm = f"{role}_bias_{i}"
                em.precomputed[nm] = env[nd]
                em.flash_bias.setdefault(anchor, {})[role] = nm

    out_node = None
    for n in gm.graph.nodes:
        if n.op == "call_function":
            em.handle(n, gm)
        elif n.op == "output":
            out_node = n.args[0]

    # fail-closed: an unknown op silently became an identity before; now we
    # stop with the full gap list so the user knows exactly which kernels to add.
    if strict and em.unsupported:
        raise UnsupportedOp(em.unsupported)

    out_val = em.ssa(out_node[0] if isinstance(out_node, (tuple, list)) else out_node)
    out_ty = _bf16_ty(em.shape[out_node[0] if isinstance(out_node,(tuple,list)) else out_node])

    body = "\n".join(em.lines)
    mlir = (f"module {{\n  func.func @{func_name}("
            + ", ".join(arg_decls) + f") -> {out_ty} {{\n"
            + body + f"\n    return {out_val} : {out_ty}\n  }}\n}}\n")

    # input SSA name -> real input tensor (so the executor can bind the activation)
    em.input_ssa = {em.val[n]: em.node2val[n] for n in placeholders
                    if em.node2val.get(n) is not None and n in em.val}
    return mlir, em, argvals


class SwinBlock(nn.Module):
    """Real Swin W-MSA block using F.scaled_dot_product_attention (HF-style)."""
    def __init__(self, dim=96, heads=3, ws=7, mlp=4):
        super().__init__()
        self.ws, self.heads, self.dim = ws, heads, dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp)
        self.fc2 = nn.Linear(dim * mlp, dim)

    def forward(self, x):                  # x: [B, H, W, C]
        B, H, W, C = x.shape
        ws, nh = self.ws, self.heads
        sc = x
        x = self.norm1(x)
        x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        nw = x.shape[0]
        qkv = self.qkv(x).reshape(nw, ws * ws, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.permute(0, 2, 1, 3).reshape(nw, ws * ws, C)
        a = self.proj(a)
        a = a.reshape(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        x = sc + a
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


if __name__ == "__main__":
    m = SwinBlock().eval()
    mlir, em, _ = lower(m, (torch.randn(1, 56, 56, 96),))
    print(mlir)
    print("// ---- lowering report ----")
    print("// folded shape-only / unhandled nodes:")
    from collections import Counter
    for k, c in Counter(em.folded).most_common():
        print(f"//   {c:3d}  {k}")
