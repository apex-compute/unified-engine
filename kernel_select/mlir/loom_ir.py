#!/usr/bin/env python3
"""loom_ir.py — the GENERIC IR-DRIVEN executor.

This is the compiler back-end done right: it does NOT hardcode Swin's structure.
It WALKS the ue MLIR op stream produced by torch_to_ue and executes each op on the
FPGA, binding the real per-op weights the frontend recorded (op_weights). The only
"structure" it knows is two generic idioms common to vision transformers:
  * the windowing idiom  (reshape, permute[0,1,3,2,4,5], reshape) -> window_partition/reverse
  * the attention idiom  (qkv matmul -> reshape/permute/select -> flash -> permute/reshape -> proj)
Everything else (layer_norm, matmul, eltwise, views) dispatches op-by-op.

Same engine for any model whose ops are in the vocabulary — Swin, ViT, SigLIP.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

import torch_to_ue
from loom_walker import parse_mlir
from loom_exec import (_alloc, _alloc_param, _pbi_set, _ln_chunks, _sram_add,
                       emit_windowed_attention)
from validate_attention import snr_db
# Generic DRAM data-movement kernels come from the neutral kernel library
# (nn_lib), NOT from any model's reference test file. The executor resolves
# op-name -> kernel; it has no per-model dependency.
from nn_lib import (window_partition_dram, window_reverse_dram,
                    cyclic_shift_dram, patch_merging_gather_dram)
from loom_plan import AttentionSpec
from user_dma_core import UE_VECTOR_SIZE

WIN_PERM = "[0, 1, 3, 2, 4, 5]"          # the window partition/reverse permute signature


def _MN(shape):
    """ND shape -> (M = prod(leading), N = last)."""
    import math
    return int(math.prod(shape[:-1])) if len(shape) > 1 else 1, shape[-1]


def _trace_back_matmul(ssa, producer):
    """Follow single-operand view ops (select/permute/reshape) back to a matmul."""
    seen = []
    while ssa in producer:
        o = producer[ssa]
        seen.append(o)
        if o["op"] == "matmul":
            return o, seen
        if o["op"] in ("select", "permute", "reshape"):
            ssa = o["operands"][0]
        else:
            break
    return None, seen


def _trace_fwd_matmul(ssa, consumers):
    """Follow view ops forward from an SSA value to the matmul that consumes it."""
    seen = []
    while ssa in consumers:
        c = consumers[ssa]
        seen.append(c)
        if c["op"] == "matmul":
            return c, seen
        if c["op"] in ("permute", "reshape"):
            ssa = c["res"]
        else:
            break
    return None, seen


class IRExecutor:
    """Walk ue MLIR ops, bind real weights, execute the whole graph on the FPGA."""

    def __init__(self, model, example_inputs, *, ws=12, patch_spec=None, strict=True):
        self.mlir, self.em, self.argvals = torch_to_ue.lower(
            model, example_inputs, patch_spec=patch_spec, strict=strict)
        self.args, self.ops = parse_mlir(self.mlir)
        self.ow = self.em.op_weights          # ssa -> {weight,bias,gamma,beta}
        self.in_ssa = self.em.input_ssa       # arg ssa -> real input tensor
        self.ws = ws
        self.model, self.example_inputs = model, example_inputs
        self._analyze()

    # ---- idiom recognition (generic; no Swin hardcode) --------------------
    def _analyze(self):
        producer = {o["res"]: o for o in self.ops}
        consumers = {}
        for o in self.ops:
            for x in o["operands"]:
                consumers[x] = o
        self.consumed = set()                 # ssa results subsumed by a macro
        self.attn = {}                        # proj_mm.res -> attention descriptor
        self.window = {}                       # last-reshape.res -> ("partition"|"reverse", input_ssa)
        used = set()
        for o in self.ops:
            for x in o["operands"]:
                used.add(x)
        # dead view ops (e.g. the leftover K-transpose the eager-attention collapse
        # leaves dangling) reference span-internal SSAs -> drop them.
        for o in self.ops:
            if o["op"] in ("reshape", "select", "permute") and o["res"] not in used:
                self.consumed.add(o["res"])

        # attention spans: anchor on flash, find qkv matmul (back) + proj matmul (fwd)
        seen_flash = 0
        for o in self.ops:
            if o["op"] != "flash_attn_pf":
                continue
            # trace back EACH of the q/k/v operands to its OWN matmul. Real HF Swin
            # has SEPARATE q/k/v linears (3 distinct matmuls sharing one input); the
            # demo SwinBlock has one FUSED qkv linear (all 3 trace to the same matmul).
            # We capture all three SSAs and let _attn_weights handle both cases.
            allback = []
            qkv_mms = []
            for opnd in o["operands"][:3]:
                mm, bk = _trace_back_matmul(opnd, producer)
                allback += bk
                qkv_mms.append(mm)
            proj_mm, fwd = _trace_fwd_matmul(o["res"], consumers)
            if not (all(qkv_mms) and proj_mm):
                continue
            qsh = next(p["out"] for p in self.ops if p["res"] == o["operands"][0])
            nw, nh, wa, hd = qsh[0], qsh[1], qsh[2], qsh[3]
            self.attn[proj_mm["res"]] = dict(
                windowed_in=qkv_mms[0]["operands"][0],
                q_ssa=qkv_mms[0]["res"], k_ssa=qkv_mms[1]["res"], v_ssa=qkv_mms[2]["res"],
                proj_ssa=proj_mm["res"], nh=nh, hd=hd, wa=wa, nw=nw, bias_idx=seen_flash)
            # consume everything from qkv matmul through proj matmul (exclusive of out)
            for p in allback + [o] + fwd:
                self.consumed.add(p["res"])
            self.consumed.discard(proj_mm["res"])   # proj_mm.res is where we EMIT
            seen_flash += 1

        # window idiom: reshape, permute(rank-6), reshape — first=partition, last=reverse.
        # (rank-6 permute is the window-axis swap signature; parse mangles the perm
        # list attr, so key on rank instead — robust and unambiguous in Swin.)
        # partition out = [1, G, G, ws, ws, C]  -> out[1]==out[2]  (grid dims adjacent)
        # reverse   out = [1, G, ws, G, ws, C]  -> out[1]==out[3]  (grid dims split)
        win_perms = [o for o in self.ops if o["op"] == "permute"
                     and o["out"] and len(o["out"]) == 6]
        for p in win_perms:
            pre = producer.get(p["operands"][0])       # the reshape before
            post = consumers.get(p["res"])             # the reshape after
            if not (pre and post and pre["op"] == "reshape" and post["op"] == "reshape"):
                continue
            out = p["out"]
            kind = "partition" if out[1] == out[2] else "reverse"
            inp = pre["operands"][0]                    # real input to the window op
            self.window[post["res"]] = (kind, inp)
            for q in (pre, p):
                self.consumed.add(q["res"])            # post.res is where we EMIT

    # ---- weight binding ---------------------------------------------------
    def _attn_weights(self, ue, desc, dim):
        nh, hd = desc["nh"], desc["hd"]
        hd_pad = ((hd + 63) // 64) * 64
        w = {}
        if desc["q_ssa"] == desc["k_ssa"] == desc["v_ssa"]:
            # FUSED qkv (demo): one [3dim,dim] weight -> slice into q/k/v.
            wA = self.ow[desc["q_ssa"]]
            qkv_w, qkv_b = wA["weight"], wA["bias"]
            for i, nm in enumerate(("q", "k", "v")):
                w[f"{nm}_weight"] = _alloc_param(ue, qkv_w[i * dim:(i + 1) * dim])
                w[f"{nm}_bias"] = _alloc_param(ue, qkv_b[i * dim:(i + 1) * dim])
        else:
            # SEPARATE q/k/v linears (real HF): bind each [dim,dim] weight directly.
            for nm, key in (("q", "q_ssa"), ("k", "k_ssa"), ("v", "v_ssa")):
                ow = self.ow[desc[key]]
                w[f"{nm}_weight"] = _alloc_param(ue, ow["weight"])
                w[f"{nm}_bias"] = _alloc_param(ue, ow["bias"])
        pA = self.ow[desc["proj_ssa"]]
        w["out_weight"] = _alloc_param(ue, pA["weight"])
        w["out_bias"] = _alloc_param(ue, pA["bias"])
        unpad = torch.zeros(dim, nh * hd_pad)
        for h in range(nh):
            for d in range(hd):
                unpad[h * hd + d, h * hd_pad + d] = 1.0
        w["unpad_weight"] = _alloc_param(ue, unpad)
        return w

    def _attn_bias(self, desc, wa_pad):
        """Build the flash bias matrix [nw*nh, wa_pad, wa_pad] from the frontend's
        precomputed relpos + shifted-window mask. Batch order is window-major,
        head-inner (matches multihead_pad_and_permute): bias[w*nh+h]=relpos[h]+mask[w].
        Padded key columns get -1e4 so they don't pollute the softmax denominator."""
        nw, nh, wa = desc["nw"], desc["nh"], desc["wa"]
        i = desc["bias_idx"]
        rp = torch.zeros(nh, wa, wa)             # per-head relpos (broadcast over windows)
        mk = torch.zeros(nw, wa, wa)             # per-window mask  (broadcast over heads)
        comb = None                              # already-combined [nw*nh, wa, wa]
        # Classify each precomputed bias for this span by SHAPE, not by the recognizer's
        # label (the eager add order can swap relpos/mask naming): leading dim nh=relpos,
        # nw=mask, nw*nh=already combined.
        for key in (f"relpos_bias_{i}", f"mask_bias_{i}"):
            t = self.em.precomputed.get(key)
            if t is None:
                continue
            flat = t.float().reshape(-1, wa, wa)
            lead = flat.shape[0]
            if lead == nh and nh != nw:
                rp = flat
            elif lead == nw and nh != nw:
                mk = flat
            elif lead == nw * nh:
                comb = flat if comb is None else comb + flat
            elif lead == nh:                     # nh==nw tie: relpos slot wins
                rp = flat
            elif lead == nw:
                mk = flat
            else:
                raise ValueError(f"bias {key} lead {lead} != nh{nh}/nw{nw}/{nw*nh}")
        bias = torch.zeros(nw * nh, wa_pad, wa_pad)
        bias[:, :, wa:] = -1e4
        for wi in range(nw):
            for h in range(nh):
                base = comb[wi * nh + h] if comb is not None else (rp[h] + mk[wi])
                bias[wi * nh + h, :wa, :wa] = base
        return bias

    # ---- host/device boundary detection -----------------------------------
    def encoder_span(self):
        """Generic host/device split for a vision encoder: the host computes the
        patch-embedding (up to the embeddings LayerNorm) and the head (final
        LayerNorm + pool + classifier); the DEVICE runs the transformer encoder
        in between. Returns (enc_in_ssa, enc_out_ssa):
          enc_in_ssa  = first layer_norm's result  (= embeddings output; host feeds it)
          enc_out_ssa = input to the last layer_norm (= encoder sequence output; host takes it)"""
        norms = [o for o in self.ops if o["op"] == "layer_norm"]
        if len(norms) < 2:
            raise ValueError("encoder_span: need >=2 layer_norms (embed + head)")
        return norms[0]["res"], norms[-1]["operands"][0]

    # ---- the walk ---------------------------------------------------------
    def run(self, banner=True, inject=None, stop_after=None):
        """inject=(ssa, tensor): preload a host-computed activation at ssa and skip
        all ops up to it (the host region). stop_after=ssa: last device op; ops
        after it are the host head and are skipped."""
        from user_dma_core import UnifiedEngine
        # Rebase the DRAM map for full-model scale: the default tensor-scratch
        # region (0xB0000000..0xD0000000 = 512MB) overflows for an 830-op encoder
        # whose stage-0 tensors are 9216 tokens. Push the program base up to
        # 0xFC000000 so scratch gets ~1.3GB; keep everything < 0x100000000 (the
        # 4GB-map contract). params stays at 0x80000000 (default, ~700MB headroom).
        ue = UnifiedEngine(tensor_dram_base=0xB2000000, program_dram_base=0xF0000000)
        sym = {}                                       # ssa -> dram addr
        shp = {}                                       # ssa -> shape
        idx = {o["res"]: i for i, o in enumerate(self.ops)}
        inject_idx = idx[inject[0]] if inject else -1
        stop_idx = idx[stop_after] if stop_after else len(self.ops)
        if inject is not None:
            inj_ssa, inj_val = inject
            v = inj_val.reshape(_MN(tuple(inj_val.shape)))
            sym[inj_ssa] = _alloc_param(ue, v.to(torch.bfloat16))
            shp[inj_ssa] = list(inj_val.shape)
        else:
            for a_ssa, val in self.in_ssa.items():
                v = val.reshape(_MN(tuple(val.shape))) if val.dim() > 2 else val
                sym[a_ssa] = _alloc_param(ue, v.to(torch.bfloat16))
                shp[a_ssa] = list(val.shape)
        gpr_mm, gpr_ln, gpr_pbi = ue.alloc_isa_reg(), ue.alloc_isa_reg(), ue.alloc_isa_reg()
        alloc = lambda n: _alloc(ue, n)

        prog = ue.get_program_dram_addr()
        ue.clear_inst_id(); ue.start_capture()
        last = None
        for i, o in enumerate(self.ops):
            res, op = o["res"], o["op"]
            shp[res] = o["out"]
            if i <= inject_idx or i > stop_idx:        # host region (embed / head)
                continue
            if res in self.consumed:
                continue
            # --- attention macro (emit at proj matmul position) ---
            if res in self.attn:
                d = self.attn[res]
                M, dim = _MN(o["out"])
                spec = AttentionSpec(kind="windowed", num_q_heads=d["nh"],
                                     num_kv_heads=d["nh"], head_dim=d["hd"],
                                     seq_len=d["wa"], batch=d["nw"], bias="none")
                w = self._attn_weights(ue, d, dim)
                wa_pad = spec.seq_pad
                bias = self._attn_bias(d, wa_pad)
                consts = {"identity": _alloc_param(ue, torch.eye(UE_VECTOR_SIZE)),
                          "bias": _alloc_param(ue, bias)}
                sym[res] = emit_windowed_attention(
                    ue, spec, in_addr=sym[d["windowed_in"]], w=w, bias_addr=consts["bias"],
                    identity_addr=consts["identity"], gpr_mm=gpr_mm, gpr_pbi=gpr_pbi, alloc=alloc)
                last = res; continue
            # --- window partition / reverse ---
            if res in self.window:
                kind, inp = self.window[res]
                ish = shp[inp]; C = ish[-1]
                M = 1
                for d_ in ish[:-1]:
                    M *= d_
                side = int(round(M ** 0.5))
                out = alloc(M * C)
                fn = window_partition_dram if kind == "partition" else window_reverse_dram
                fn(ue, INPUT_DRAM_ADDR=sym[inp], OUTPUT_DRAM_ADDR=out,
                   H=side, W=side, C=C, window_size=self.ws)
                sym[res] = out; last = res; continue
            # --- cyclic shift (torch.roll on a spatial [1,H,W,C] map) ---
            if op == "roll":
                inp = o["operands"][0]; ish = shp[inp]
                H, W, C = ish[1], ish[2], ish[3]
                sv = int(str(o["attrs"].get("shifts", "0")).strip("[]"))   # signed shift
                sh, sw = (-sv) % H, (-sv) % W       # torch.roll(sv) == cyclic_shift(-sv)
                out = alloc(H * W * C)
                cyclic_shift_dram(ue, INPUT_DRAM_ADDR=sym[inp], OUTPUT_DRAM_ADDR=out,
                                  H=H, W=W, C=C, shift_h=sh, shift_w=sw)
                sym[res] = out; last = res; continue
            # --- patch merge: 2x2 gather only (the following LN + matmul reduce) ---
            if op == "patch_merge":
                inp = o["operands"][0]; ish = shp[inp]
                M = 1
                for d_ in ish[:-1]:
                    M *= d_
                C = ish[-1]; side = int(round(M ** 0.5))
                out = alloc((side // 2) * (side // 2) * 4 * C)
                patch_merging_gather_dram(ue, INPUT_DRAM_ADDR=sym[inp],
                                          OUTPUT_DRAM_ADDR=out, H=side, W=side, C=C)
                sym[res] = out; last = res; continue
            # --- views: alias (skip dead/span-internal views) ---
            if op in ("reshape", "select", "permute"):
                src = o["operands"][0]
                if src in sym:
                    sym[res] = sym[src]; last = res
                continue
            # --- layer_norm ---
            if op == "layer_norm":
                M, N = _MN(o["out"]); ow = self.ow[res]
                g = _alloc_param(ue, ow["gamma"]); b = _alloc_param(ue, ow["beta"])
                out = alloc(M * N)
                _pbi_set(ue, gpr_ln, _ln_chunks(M, N))
                ue.layer_norm_core_dram(M=M, N=N, A_DRAM_ADDR=sym[o["operands"][0]],
                                        OUTPUT_DRAM_ADDR=out, GAMMA_DRAM_ADDR=g,
                                        BETA_DRAM_ADDR=b, gpr_M_reg=gpr_ln)
                sym[res] = out; last = res; continue
            # --- matmul (MLP fc1/fc2; act gelu) ---
            if op == "matmul":
                M, N = _MN(o["out"]); _, K = _MN(shp[o["operands"][0]])
                ow = self.ow[res]
                wB = _alloc_param(ue, ow["weight"])
                cB = _alloc_param(ue, ow["bias"]) if ow.get("bias") is not None else None
                gelu = o["attrs"].get("act", "").strip('"') == "gelu"
                out = alloc(M * N)
                _pbi_set(ue, gpr_mm, M)
                ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=sym[o["operands"][0]],
                                   B_DRAM_ADDR=wB, OUTPUT_DRAM_ADDR=out, C_DRAM_ADDR=cB,
                                   bias_mode="broadcast_N" if cB else "broadcast_N",
                                   gelu_enable=gelu, gpr_M_reg=gpr_mm)
                sym[res] = out; last = res; continue
            # --- eltwise add (residual) ---
            if op == "eltwise":
                M, N = _MN(o["out"]); out = alloc(M * N)
                _sram_add(ue, sym[o["operands"][0]], sym[o["operands"][1]], out, M * N)
                sym[res] = out; last = res; continue
            raise ValueError(f"IRExecutor: unhandled op '{op}' ({res})")

        ue.generate_instruction_halt(); ue.stop_capture()
        size = ue.write_captured_instructions_to_dram(prog)
        ue.allocate_program_dram(size)
        if banner:
            print(f"[ir] walked {len(self.ops)} ops -> {size/1024:.1f} KB program "
                  f"({len(self.attn)} attention, {len(self.window)} window ops)")
        print(f"[ir] DRAM usage: params={ue.get_params_dram_usage()/1e6:.0f}MB "
              f"tensor={ue.get_tensor_dram_usage()/1e6:.0f}MB (limit "
              f"{(0xF0000000-0xB2000000)/1e6:.0f}MB) program={size/1e6:.1f}MB")
        ue.program_execute(prog)
        M, N = _MN(shp[last])
        hw = ue.dma_from_accelerator_memory(sym[last], (M, N)).float()
        return hw, last


def validate(dim=192, heads=6, ws=12, H=12):
    torch.manual_seed(0)
    blk = torch_to_ue.SwinBlock(dim=dim, heads=heads, ws=ws).eval()
    x = torch.randn(1, H, H, dim)
    ref = blk(x).reshape(H * H, dim)
    ex = IRExecutor(blk, (x,), ws=ws)
    hw, last = ex.run()
    s = snr_db(ref, hw[:, :dim])
    print(f"[ir] IR-driven Swin block  SNR = {s:.2f} dB  [{'PASS' if s > 19 else 'FAIL'}]")
    print(f"     ref[0,:4] = {[round(v,4) for v in ref[0,:4].tolist()]}")
    print(f"     hw [0,:4] = {[round(v,4) for v in hw[0,:4].tolist()]}")
    return s


def run_real(model_id="microsoft/swin-large-patch4-window12-384-in22k",
             image_path=None, ws=None):
    """End-to-end real HF Swin: HOST patch-embed -> DEVICE encoder -> HOST head.
    Compares the device-driven prediction to the full torch model (the oracle)."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id, torch_dtype=torch.float32).eval()
    cfg = model.config
    ws = ws or int(getattr(cfg, "window_size", 7))
    swin = model.swin

    img = Image.open(image_path).convert("RGB")
    px = proc(img, return_tensors="pt")["pixel_values"]

    # torch oracle (full model)
    with torch.no_grad():
        ref_logits = model(px).logits[0]
    ref_top = int(ref_logits.argmax())

    # HOST: patch-embed (conv + embeddings LayerNorm)
    with torch.no_grad():
        emb, dims = swin.embeddings(px)                # [1, L, C]
        enc_ref = swin.encoder(emb, dims).last_hidden_state   # torch encoder oracle

    # DEVICE: encoder span only
    ex = IRExecutor(swin, (px,), ws=ws, strict=False)
    enc_in, enc_out = ex.encoder_span()
    print(f"[real] model={model_id}")
    print(f"[real] ws={ws}  device encoder span {enc_in} .. {enc_out}  "
          f"({len(ex.attn)} attn, {len(ex.window)} window, "
          f"{sum(o['op']=='roll' for o in ex.ops)} roll, "
          f"{sum(o['op']=='patch_merge' for o in ex.ops)} merge)")
    hw, _ = ex.run(inject=(enc_in, emb), stop_after=enc_out)
    M, N = hw.shape
    seq_out = hw.reshape(1, M, N)
    enc_snr = snr_db(enc_ref.reshape(M, N), seq_out.reshape(M, N))
    print(f"[real] ENCODER SNR (device vs torch) = {enc_snr:.2f} dB  "
          f"[{'PASS' if enc_snr > 19 else 'FAIL'}]")

    # HOST: head (final LayerNorm + mean-pool + classifier)
    with torch.no_grad():
        seq = swin.layernorm(seq_out)
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            seq.transpose(1, 2), 1).flatten(1)
        logits = model.classifier(pooled)[0]
    hw_top = int(logits.argmax())

    id2label = cfg.id2label
    match = "MATCH" if hw_top == ref_top else "MISMATCH"
    print(f"[real] torch  top = {ref_top:>6}  {id2label.get(ref_top)}")
    print(f"[real] device top = {hw_top:>6}  {id2label.get(hw_top)}   [{match}]")
    return hw_top, ref_top


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="run a real HF Swin model")
    ap.add_argument("--model", default="microsoft/swin-large-patch4-window12-384-in22k")
    ap.add_argument("--image", default="../../test_samples/vette.jpg")
    ap.add_argument("--ws", type=int, default=None)
    args = ap.parse_args()
    if args.real:
        run_real(args.model, args.image, args.ws)
    else:
        validate()
