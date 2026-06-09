"""hlo.py — THE HLO CATALOG: declarative mapping from op-name -> real engine kernel.

These are the high-level operation (HLO) mappings: the public registry of every op
`chain(...)` can use. To see what ops are available, read the `UE_HLO` dict below; to add one,
append a `KernelSpec` row — no new wrapper code needed.

This is the "small class that taps into user_dma_core and maintains a mapping."
Instead of hand-writing a wrapper per op, every supported kernel is one KernelSpec
row: which engine callable to invoke, whether it supports PBI (i.e. accepts a
gpr_M_reg), how to build its bespoke kwargs from (inputs, output, attrs), and how to
size its output. The dispatcher allocates the output, picks legacy/PBI, calls the
real function (which emits + captures the instruction stream), and returns a Tensor.

PBI MODEL (uniform across the engine): every `*_core_dram` / matmul / attention base
function is a DISPATCHER — pass a `gpr_M_reg` (a primed GPR index) and it routes to
the `_pbi` variant; omit it and it runs `_legacy`. So "support PBI" = "pass a
register or not." The decision comes from: explicit attrs {"pbi": True/False}, else
the stage policy (prefill/decode) the host carries.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from ir_harness import pad64


@dataclass
class KernelSpec:
    name: str
    fn: str                              # attribute name on the engine (ue) or nn_lib
    src: str = "ue"                      # "ue" (UnifiedEngine method) | "nn" (nn_lib func)
    pbi: bool = False                    # base fn accepts gpr_M_reg -> PBI capable
    build: Callable = None               # (host, ins, out, attrs, reg) -> kwargs dict
    out_shape: Callable = None           # (ins, attrs) -> logical output shape
    # for nn_lib funcs the first positional arg is `ue`; build returns kwargs only,
    # the dispatcher prepends ue automatically when src == "nn".


# =============================================================================
# Dispatcher — the one call site. Resolves mode, allocs output, invokes the real
# engine function (which captures the instruction stream), returns the output Tensor.
# `host` must expose: .ue, .arena, ._resolve(x), ._mode_reg(spec, attrs).
# =============================================================================
def dispatch(host, spec: KernelSpec, ins, out_slot, attrs):
    out = host._bind_out(out_slot, spec.out_shape(ins, attrs), ins[0].dtype)
    reg = host._mode_reg(spec, attrs)
    kwargs = spec.build(host, ins, out, attrs, reg)
    if spec.src == "nn":
        import nn_lib
        getattr(nn_lib, spec.fn)(host.ue, **kwargs)
    else:
        getattr(host.ue, spec.fn)(**kwargs)
    # Stage-4 mapping recorder (ground truth): one row per resolved op->engine kernel.
    mlog = getattr(host, "_map_log", None)
    if mlog is not None:
        # weight precision: any non-bf16 input tensor is the quantized weight (its dtype IS
        # the precision the kernel runs at — matmat_mul_core sets is_B_quantized/data_type
        # from it, so a quantized prefill matmul keeps the `matmul`/matmat_mul_core name).
        prec = next((i.dtype for i in ins if getattr(i, "dtype", "bf16") != "bf16"), "bf16")
        mlog.append({
            "program": getattr(host, "_cur_program", None),
            "ctx": getattr(host, "_map_ctx", None),
            "op": spec.name, "kernel": spec.fn, "src": spec.src,
            "pbi_capable": spec.pbi, "prec": prec,
            "mode": "legacy" if reg is None else f"PBI(r{int(reg)})",
            "fused": None, "out": getattr(out, "name", ""),
        })
    return out


# =============================================================================
# UE_HLO — the mapping. Add a row to support an op; no new wrapper code.
# =============================================================================
def _mn(t):            # (M, N) padded
    return pad64(t.shape[0]), pad64(t.shape[1])


def _mode(m):          # accept UE_MODE or a string; return the real UE_MODE enum
    from user_dma_core import UE_MODE
    if isinstance(m, str):
        return {"ADD": UE_MODE.ELTWISE_ADD, "MUL": UE_MODE.ELTWISE_MUL}[m.upper()]
    return m


def _type(prec):       # weight precision string -> hardware TYPE for the decode path
    from user_dma_core import TYPE
    table = {"if4": TYPE.IF4, "if8": TYPE.IF8}
    if prec not in table:
        raise ValueError(f"no TYPE mapping for precision '{prec}' "
                         f"(supported: {list(table)}); quantize with one of these")
    return table[prec]


def _build_matmul(h, ins, out, a, reg):
    """matmat_mul_core kwargs. Computes A @ Bᵀ (HF Linear convention): the weight B is
    [out_features, in_features], so K = A.cols = B.cols and N = B.rows. Pass HF weights
    directly — no transpose needed. Auto-quant: if B is not bf16, set is_B_quantized +
    data_type + SCALE from the weight handle's own .scale buffer."""
    A, B = ins[0], ins[1]
    # M is the real row count. Single-token decode (M=1) must stay 1, not pad64(1)=64,
    # or every decode matmul does 64x the work (cores accept any M; reference uses M=1).
    M = A.shape[0] if A.shape[0] == 1 else pad64(A.shape[0])
    kw = dict(
        M=M, K=pad64(A.shape[1]), N=pad64(B.shape[0]),
        A_DRAM_ADDR=A.addr, B_DRAM_ADDR=B.addr, OUTPUT_DRAM_ADDR=out.addr,
        C_DRAM_ADDR=(h._resolve(a["bias"]).addr if a.get("bias") else None),
        gelu_enable=(a.get("act") == "gelu"), silu_enable=(a.get("act") == "silu"),
        # lm_head: skip the 262k-logit write-back; the engine still tracks the argmax
        # register (read via ue.get_arg_max_index()), so token pick needs no host matmul.
        write_back_disable=a.get("write_back_disable", False),
        gpr_M_reg=reg,
    )
    if B.dtype != "bf16":
        if B.scale is None:
            raise ValueError(f"quantized weight '{B.name}' has no .scale buffer "
                             f"(load it via lower_precision + load_weights)")
        kw.update(is_B_quantized=True, data_type=_type(B.dtype), SCALE_DRAM_ADDR=B.scale.addr)
    return kw


UE_HLO: dict[str, KernelSpec] = {

    "rms_norm": KernelSpec(
        name="rms_norm", fn="rms_norm_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            A_DRAM_ADDR=ins[0].addr, OUTPUT_DRAM_ADDR=out.addr,
            GAMMA_DRAM_ADDR=ins[1].addr, gpr_M_reg=reg),
    ),

    "layer_norm": KernelSpec(
        name="layer_norm", fn="layer_norm_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            A_DRAM_ADDR=ins[0].addr, OUTPUT_DRAM_ADDR=out.addr,
            GAMMA_DRAM_ADDR=ins[1].addr, BETA_DRAM_ADDR=ins[2].addr, gpr_M_reg=reg),
    ),

    "matmul": KernelSpec(   # A @ Bᵀ, B is [out,in] (HF). bf16 or quantized (auto via B.dtype)
        name="matmul", fn="matmat_mul_core", pbi=True,
        out_shape=lambda ins, a: (ins[0].shape[0], ins[1].shape[0]),
        build=_build_matmul,
    ),

    "qmatmul": KernelSpec(   # decode-time quantized matmul (M=1). A @ Bᵀ, B is [out,in].
        name="qmatmul", fn="quantized_matmat_core", pbi=False,
        out_shape=lambda ins, a: (ins[0].shape[0], ins[1].shape[0]),
        build=lambda h, ins, out, a, reg: dict(
            M=(ins[0].shape[0] if ins[0].shape[0] == 1 else pad64(ins[0].shape[0])),
            K=pad64(ins[0].shape[1]), N=pad64(ins[1].shape[0]),
            A_DRAM_ADDR=ins[0].addr, B_DRAM_ADDR=ins[1].addr, OUTPUT_DRAM_ADDR=out.addr,
            SCALE_DRAM_ADDR=ins[1].scale.addr, data_type=_type(ins[1].dtype),
            C_DRAM_ADDR=(h._resolve(a["bias"]).addr if a.get("bias") else None),
            gelu_enable=(a.get("act") == "gelu"), silu_enable=(a.get("act") == "silu"),
            # lm_head: skip the logit write-back; quantized_matmat_core still updates the
            # hardware argmax register (read via get_arg_max_index) — unlike matmat_mul_core.
            write_back_disable=a.get("write_back_disable", False)),
    ),

    "rope": KernelSpec(
        name="rope", fn="rope_hf_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            input_dram_addr=ins[0].addr, output_dram_addr=out.addr,
            cos_dram_addr=ins[1].addr, sin_dram_addr=ins[2].addr, gpr_M_reg=reg),
    ),

    "rope_gqa": KernelSpec(   # input [seq, group*head_dim]; N=head_dim=width//group
        name="rope_gqa", fn="rope_hf_core_dram_gqa", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], group_size=a["group"], N=ins[0].shape[1] // a["group"],
            input_dram_addr=ins[0].addr, output_dram_addr=out.addr,
            cos_dram_addr=ins[1].addr, sin_dram_addr=ins[2].addr, gpr_M_reg=reg),
    ),

    "rope_decode": KernelSpec(   # single-token decode RoPE; rope-table row at runtime
        name="rope_decode", fn="rope_hf_core_decode", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            N=ins[0].shape[-1],
            input_dram_addr=ins[0].addr, output_dram_addr=out.addr,
            gr_weight_dram=a["gr_weight"]),   # GPR = rope_base + pos*2*head_dim*bpe
    ),

    "rope_d64": KernelSpec(   # head_dim<128 PER-HEAD rope. Each rotate-half operand is N/2 elems
        # = N bytes < 128 (half a URAM row, un-sliceable), so rope_hf_core_dram dispatches N<128 to
        # the padded-split path: register-addressed DMAs land each half on its own 128-byte URAM row.
        # Table is PACKED [cos(N) || sin(N)] contiguous per token (sin-lo pre-negated); the kernel
        # asserts sin_addr == cos_addr + N*2, so we pass the table base as cos and base+N*2 as sin.
        name="rope_d64", fn="rope_hf_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            input_dram_addr=ins[0].addr, output_dram_addr=out.addr,
            cos_dram_addr=a["table"], sin_dram_addr=a["table"] + ins[0].shape[1] * 2,
            gpr_M_reg=reg),
    ),

    "rope_decode_d64": KernelSpec(   # single-token decode rope for head_dim<128 (decode analog of
        # rope_d64). Token at runtime pos_reg is roped against the packed table row at pos*2*N*bpe.
        name="rope_decode_d64", fn="rope_hf_core_decode_d64", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            N=ins[0].shape[-1],
            input_dram_addr=ins[0].addr, output_dram_addr=out.addr,
            packed_table_addr=a["table"], pos_reg=a["pos_reg"], tmp_reg=a["tmp_reg"],
            output_pos_strided=a.get("pos_strided", False)),
    ),

    "permute": KernelSpec(   # ND reshape/transpose via DMA gather + batched identity-dot transpose.
        # Used by the vision encoder/connector: patch-embed reshape, per-head Q/K/V split + inverse,
        # and the connector pixel-shuffle. Shape changes, so caller passes the logical output shape
        # and the full ND dims+perm in attrs. Needs a scratch DRAM region (attrs["temp"]).
        name="permute", fn="smart_bf16_permute_core", src="nn", pbi=False,
        out_shape=lambda ins, a: a["out_shape"],
        build=lambda h, ins, out, a, reg: dict(
            dims=a["dims"], permute_indices=a["perm"],
            input_dram_addr=ins[0].addr, output_dram_addr=out.addr,
            temp_dram_start=a.get("temp", 0)),
    ),

    "layer_norm_post_add": KernelSpec(   # fused: layer_norm(A + B). Vision encoder post-attn/mlp
        # residual+LN in one kernel. gamma/beta both required (LayerNorm has a learnable bias).
        name="layer_norm_post_add", fn="layer_norm_core_dram_post_add", src="nn", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            A_DRAM_ADDR=ins[0].addr, B_DRAM_ADDR=ins[1].addr,
            ADDOUTPUT_DRAM_ADDR=a["sum_out"], NORMOUTPUT_DRAM_ADDR=out.addr,
            GAMMA_DRAM_ADDR=ins[2].addr, BETA_DRAM_ADDR=ins[3].addr),
    ),

    "eltwise": KernelSpec(   # generic add/mul/sub via mode in attrs["mode"]
        name="eltwise", fn="eltwise_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            dram_a=ins[0].addr, dram_b=ins[1].addr, dram_out=out.addr,
            mode=_mode(a["mode"]), gpr_M_reg=reg),
    ),

    "add": KernelSpec(       # convenience: eltwise add (mode baked)
        name="add", fn="eltwise_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            dram_a=ins[0].addr, dram_b=ins[1].addr, dram_out=out.addr,
            mode=_mode("ADD"), gpr_M_reg=reg),
    ),

    "mul": KernelSpec(       # convenience: eltwise mul (mode baked)
        name="mul", fn="eltwise_core_dram", pbi=True,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            M=ins[0].shape[0], N=ins[0].shape[1],
            dram_a=ins[0].addr, dram_b=ins[1].addr, dram_out=out.addr,
            mode=_mode("MUL"), gpr_M_reg=reg),
    ),

    "flash_attn": KernelSpec(   # gemma prefill: bucketized flash_attention_core (ue method)
        name="flash_attn", fn="flash_attention_core", src="ue", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            head_dim=ins[0].shape[-1], seq_len=ins[0].shape[0],
            Q_DRAM_ADDR=ins[0].addr, K_DRAM_ADDR=ins[1].addr, V_DRAM_ADDR=ins[2].addr,
            OUTPUT_DRAM_ADDR=out.addr,
            SCRATCH_DRAM_ADDR=h._resolve(a["scratch"]).addr,
            IDENTITY_DRAM_ADDR=(h._resolve(a["identity"]).addr if a.get("identity") else None),
            BIAS_DRAM_ADDR=(h._resolve(a["mask"]).addr if a.get("mask") else None),
            ATTN_P_DRAM_ADDR=(h._resolve(a["attn_p"]).addr if a.get("attn_p") else None),
            gpr_bucket_idx=a.get("bucket_reg"), num_buckets=a.get("num_buckets", 8)),
    ),

    "flash_attn_pf": KernelSpec(   # smolvlm-style non-bucketed prefill_flash_attention_core (nn)
        name="flash_attn_pf", fn="prefill_flash_attention_core", src="nn", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            head_dim=ins[0].shape[-1], seq_len=ins[0].shape[0],
            Q_DRAM_ADDR=ins[0].addr, K_DRAM_ADDR=ins[1].addr, V_DRAM_ADDR=ins[2].addr,
            OUTPUT_DRAM_ADDR=out.addr,
            SCRATCH_DRAM_ADDR=h._resolve(a["scratch"]).addr,
            IDENTITY_DRAM_ADDR=h._resolve(a["identity"]).addr,
            BIAS_DRAM_ADDR=h._resolve(a["mask"]).addr,
            num_q_heads=a["group"]),
    ),

    "group_attn": KernelSpec(   # gemma decode: bucketized decoder_group_attention_core
        name="group_attn", fn="decoder_group_attention_core", src="ue", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            group_size=a["group"], head_dim=ins[0].shape[-1],
            # bucket_reg=None -> legacy (single static seq_len body, no PBI bucket dispatcher).
            # seq_len_override lets the legacy path read a fixed KV window (e.g. MAXPOS) and
            # mask via the bias — avoids back-to-back PBI corruption + 8x program bloat.
            seq_len=a.get("seq_len_override", ins[0].shape[0]),
            Q_DRAM_ADDR=ins[0].addr, K_DRAM_ADDR=ins[1].addr, V_DRAM_ADDR=ins[2].addr,
            OUTPUT_DRAM_ADDR=out.addr,
            SCRATCH_DRAM_ADDR=h._resolve(a["scratch"]).addr,
            IDENTITY_DRAM_ADDR=(h._resolve(a["identity"]).addr if a.get("identity") else None),
            BIAS_DRAM_ADDR=(h._resolve(a["mask"]).addr if a.get("mask") else None),
            gpr_bucket_idx=a.get("bucket_reg"), num_buckets=a.get("num_buckets", 8)),
    ),

    "group_attn_pbi": KernelSpec(   # smolvlm2 decode: bucketized decoder_group_attention_core_PBI.
        # Distinct engine fn from group_attn: a JZ-cascade dispatcher emits num_buckets bucket
        # bodies (seq_len = 64*i) and routes via the runtime gpr_bucket_idx (1-based). Q/K/V are
        # head-major: Q=[group,head_dim] (the kv-group's roped Q heads), K/V = the per-(layer,head)
        # cache base holding all positions contiguous. use_pbi=False keeps the inner flash body
        # legacy (verified smolvlm2 setting — PBI-flash back-to-back corrupts; matmul/norm/rope
        # stay PBI elsewhere). gpr_bucket_idx must be a GPR in [1,15]; num_buckets = ceil(maxctx/64).
        name="group_attn_pbi", fn="decoder_group_attention_core_pbi", src="ue", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            group_size=a["group"], head_dim=ins[0].shape[-1],
            Q_DRAM_ADDR=ins[0].addr, K_DRAM_ADDR=ins[1].addr, V_DRAM_ADDR=ins[2].addr,
            OUTPUT_DRAM_ADDR=out.addr,
            SCRATCH_DRAM_ADDR=h._resolve(a["scratch"]).addr,
            IDENTITY_DRAM_ADDR=(h._resolve(a["identity"]).addr if a.get("identity") else None),
            BIAS_DRAM_ADDR=(h._resolve(a["mask"]).addr if a.get("mask") else None),
            gpr_bucket_idx=a["bucket_reg"], num_buckets=a.get("num_buckets", 8),
            use_pbi=a.get("use_pbi", False)),
    ),

    "patching": KernelSpec(     # vision encoder: im2col + quantized matmul for patch embedding
        name="patching", fn="patching_core", src="ue", pbi=False,
        out_shape=lambda ins, a: (a["H"]//a["patch_h"] * a["W"]//a["patch_w"], a["N"]),
        build=lambda h, ins, out, a, reg: dict(
            INPUT_DRAM_ADDR=ins[0].addr, OUTPUT_DRAM_ADDR=out.addr,
            matrix_dram_addrs=a["matrix_addrs"],   # list of int addrs, one per patch-position in group
            scale_dram_addrs=a["scale_addrs"],     # list of int addrs, one per patch-position in group
            C=a["C"], H=a["H"], W=a["W"],
            patch_h=a["patch_h"], patch_w=a["patch_w"],
            K=a["K"], N=a["N"],
            data_type=_type(a.get("dtype", "if4"))),
    ),

    # ---- data movement (SRAM<->DRAM). Not chainable tensor ops — addresses only.
    # Used for inter-layer residual carry and GQA staging. Call via host helpers
    # (sel.load_sram / sel.store_dram), not through chain(), since they produce no
    # new Tensor. Registered here for discoverability / one place per op.
    "load_sram": KernelSpec(    # accelerator_memory_to_sram
        name="load_sram", fn="accelerator_memory_to_sram", src="ue", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            accelerator_dram_address=ins[0].addr, sram_address=a["sram"],
            element_size=a["element_size"]),
    ),

    "store_dram": KernelSpec(   # sram_to_accelerator_memory
        name="store_dram", fn="sram_to_accelerator_memory", src="ue", pbi=False,
        out_shape=lambda ins, a: ins[0].shape,
        build=lambda h, ins, out, a, reg: dict(
            sram_address=a["sram"], accelerator_dram_address=ins[0].addr,
            element_size=a["element_size"]),
    ),
}

# ── vision encoder weight helpers ─────────────────────────────────────────────
def conv2d_to_patch_weights(w, patch_h: int, patch_w: int,
                             C: int, N: int, dtype=None):
    """Convert a Conv2D weight (N, C, patch_h, patch_w) or (N, C*P*P) into
    patches_per_group sparse matrices for patching_core.

    patching_core processes patches_per_group = UE_VECTOR_SIZE // patch_w
    patches as a group, each with its own (N, K) sparse weight, where
    K = C * patch_h * UE_VECTOR_SIZE.

    Returns: list of patches_per_group weight tensors, each (N, K).
    """
    import torch
    from user_dma_core import UE_VECTOR_SIZE  # 64
    dtype = dtype or torch.bfloat16
    if w.dim() == 4:
        w = w.reshape(N, -1)           # (N, C*patch_h*patch_w)
    assert w.shape[1] == C * patch_h * patch_w, \
        f"expected inner dim {C * patch_h * patch_w}, got {w.shape[1]}"

    ppg = UE_VECTOR_SIZE // patch_w     # patches per group (4 for patch_w=16)
    K = C * patch_h * UE_VECTOR_SIZE
    sparse = []
    for p in range(ppg):
        ws = torch.zeros(N, K, dtype=dtype)
        for row in range(C * patch_h):  # each URAM row
            for pw in range(patch_w):   # each element within the patch
                ws[:, row * UE_VECTOR_SIZE + p * patch_w + pw] = \
                    w[:, row * patch_w + pw]
        sparse.append(ws)
    return sparse