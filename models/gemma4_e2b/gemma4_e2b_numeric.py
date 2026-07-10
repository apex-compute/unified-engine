#!/usr/bin/env python3
"""Gemma4 E2B VISION-encoder numeric verification: host oracle vs FPGA.

Debug tool for the vision encoder ONLY (that is the part under investigation).

The host reference *mimics the hardware*: it runs the HF Gemma4 vision tower with
every Linear weight round-tripped through IF4 — the same block-quantization the
FPGA uses for the vision matmuls (VISION_QUANT_PRECISION) — so the quantization
gap is present in BOTH sides and the residual SNR reflects HARDWARE execution
error, not the (benign, identical-on-every-HW) quant/algorithm gap.

Each run reports the FPGA readbacks (stashed by _run_vision_encoder_fpga in
self._vis_ckpt) SNR-compared against BOTH references:
  * hostref — HF vision tower with IF4-quantized weights (mimics hardware).
  * hf      — full-precision HF vision tower (ground truth).

  A = patch_embed    [S, H]           (VIS_IO_A)
  B = encoder_out    [S, H]           (final encoder hidden state, pre-pool)
  C = image_features [N_soft, 1536]   (pooler + embed_vision projection)

Uses calculate_snr from user_dma_core, mirroring gemma3_numeric.py.

Usage:
  python gemma4_e2b_numeric.py --dev xdma1
"""
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                   # import gemma4_e2b_test
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # repo root (user_dma_core)

import torch
import torch.nn as nn
from PIL import Image

import user_dma_core
from user_dma_core import set_dma_device, calculate_snr
import quant_lib
import gemma4_e2b_test as g4
from gemma4_e2b_test import (
    Gemma4_UnifiedEngine, _ensure_hf_model,
    VISION_CANONICAL_SIZE, VISION_QUANT_PRECISION,
    DEFAULT_IMAGE, DEFAULT_AUDIO,
)
from transformers import AutoProcessor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _align(ref, res):
    """Flatten batch, match row count / shape between host ref and FPGA readback."""
    r = ref.detach().float()
    f = res.detach().float()
    if r.dim() == 3 and r.shape[0] == 1:
        r = r.squeeze(0)
    if f.dim() == 3 and f.shape[0] == 1:
        f = f.squeeze(0)
    if r.shape != f.shape:
        r = r.reshape(-1, r.shape[-1])
        f = f.reshape(-1, f.shape[-1])
        n = min(r.shape[0], f.shape[0])
        r, f = r[:n], f[:n]
    return r, f


def report(name, ref, res, row_mask=None):
    """Print SNR / rel_L2 / max|Δ| of FPGA `res` vs host `ref`.

    row_mask (bool over rows) restricts the comparison to real (non-padding)
    patches so the ~2264 zero padding rows don't dominate the norms.
    """
    r, f = _align(ref, res)
    if row_mask is not None and row_mask.numel() == r.shape[0]:
        r, f = r[row_mask], f[row_mask]
    m = torch.isfinite(r).all(dim=-1) & torch.isfinite(f).all(dim=-1)
    r, f = r[m], f[m]
    if r.numel() == 0:
        print(f"  [numeric] {name:20s} no finite overlap")
        return float('-inf')
    snr = float(calculate_snr(r, f))
    err = float(torch.linalg.vector_norm(f - r))
    sig = float(torch.linalg.vector_norm(r))
    rel = err / sig if sig > 0 else float('inf')
    maxd = float((f - r).abs().max())
    print(f"  [numeric] {name:20s} SNR={snr:7.2f} dB  rel_L2={rel:.4g}  "
          f"max|Δ|={maxd:.4g}  rows={r.shape[0]} shape={tuple(f.shape)}")
    return snr


# ---------------------------------------------------------------------------
# Host reference (HF vision tower, optionally IF4-quantized weights)
# ---------------------------------------------------------------------------
def quantize_vision_tower_(vt, precision, block=64):
    """Round-trip every 2D Linear weight in the vision tower through `precision`
    (mimics the FPGA's quantized vision weights). Modifies vt in place; returns
    the number of weights quantized."""
    n = 0
    with torch.no_grad():
        for mod in vt.modules():
            w = getattr(mod, "weight", None)
            if isinstance(mod, nn.Linear) and w is not None and w.dim() == 2 and w.shape[1] % block == 0:
                wq = w.detach().cpu().to(torch.bfloat16)
                N, K = wq.shape
                data, scale = quant_lib.quantize(precision, wq, block)
                deq = quant_lib.dequant(precision, data, scale, N, K, block)
                w.data.copy_(deq.to(w.dtype).to(w.device))
                n += 1
    return n


def _tensor(o):
    t = getattr(o, "last_hidden_state", None)
    if t is None:
        t = o[0] if isinstance(o, (tuple, list)) else o
    return t


def _add_l0_attention_layout_refs(caps, vt, pixel_position_ids):
    """Add layer-0 attention internals in the FPGA's per-head RoPE layout.

    The FPGA permutes Q/K/V channels within each head before RoPE and permutes
    o_proj input columns to consume that layout. These refs are therefore not
    the vanilla HF layout; they match the checkpoint buffers captured from DRAM.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import apply_multidimensional_rope
        L0 = vt.encoder.layers[0]
        attn = L0.self_attn
        hidden = caps.get("L0_before")
        if hidden is None:
            return
        H = hidden.shape[-1]
        HD = attn.head_dim
        NH = H // HD
        S = hidden.shape[-2]
        aligned_S = ((S + 63) // 64) * 64
        perm = torch.cat([
            torch.arange(0, 16),
            torch.arange(32, 48),
            torch.arange(16, 32),
            torch.arange(48, 64),
        ]).to(hidden.device)

        def _fpga_interleaved(x):
            # [1,S,NH,HD] standard HF order -> [S,NH*HD] FPGA per-head perm order.
            return x.squeeze(0)[:, :, perm].reshape(S, NH * HD).detach()

        def _head_major(x_interleaved, prescale=1.0):
            x = x_interleaved.reshape(S, NH, HD).transpose(0, 1).contiguous()
            out = torch.zeros((NH, aligned_S, HD), dtype=x.dtype, device=x.device)
            out[:, :S, :] = x * prescale
            return out.reshape(NH * aligned_S, HD).detach()

        cos, sin = vt.encoder.rotary_emb(caps["A"], pixel_position_ids)
        hidden_shape = (*hidden.shape[:-1], NH, HD)
        q_proj = attn.q_proj(hidden).view(hidden_shape)
        k_proj = attn.k_proj(hidden).view(hidden_shape)
        v_proj = attn.v_proj(hidden).view(hidden_shape)
        q_norm = attn.q_norm(q_proj)
        k_norm = attn.k_norm(k_proj)
        v_norm = attn.v_norm(v_proj)
        rope_q = apply_multidimensional_rope(q_norm, cos, sin, pixel_position_ids)
        rope_k = apply_multidimensional_rope(k_norm, cos, sin, pixel_position_ids)

        q_proj_i = _fpga_interleaved(q_proj)
        k_proj_i = _fpga_interleaved(k_proj)
        v_proj_i = _fpga_interleaved(v_proj)
        q_norm_i = _fpga_interleaved(q_norm)
        k_norm_i = _fpga_interleaved(k_norm)
        v_norm_i = _fpga_interleaved(v_norm)
        rope_q_i = _fpga_interleaved(rope_q)
        rope_k_i = _fpga_interleaved(rope_k)
        attn_core = caps.get("L0_attn_core")
        if attn_core is not None:
            attn_core = attn_core.reshape(1, S, NH, HD)
            attn_core_i = _fpga_interleaved(attn_core)
            caps["L0_attn_core_fpga"] = attn_core_i
            caps["L0_attn_out_hm"] = _head_major(attn_core_i)

        caps.update({
            "L0_q_proj": q_proj_i,
            "L0_k_proj": k_proj_i,
            "L0_v_proj": v_proj_i,
            "L0_q_pre_rope": q_norm_i,
            "L0_k_pre_rope": k_norm_i,
            "L0_v_norm": v_norm_i,
            "L0_rope_q": rope_q_i,
            "L0_rope_k": rope_k_i,
            "L0_q_hm": _head_major(rope_q_i, prescale=float(HD) ** 0.5),
            "L0_k_hm": _head_major(rope_k_i),
            "L0_v_hm": _head_major(v_norm_i),
        })
    except Exception as e:
        print(f"  [numeric] (layer-0 attention-layout refs unavailable: {type(e).__name__}: {e})")


def _rows(t):
    return t.shape[-2] if (t is not None and hasattr(t, "dim") and t.dim() >= 2) else -1


def build_references(cfg, image_path, prompt):
    """Run the HF vision tower TWICE and capture per-stage outputs (A, B, C):

      * "hf"      — full-precision HF vision tower (ground truth).
      * "hostref" — same HF tower with its Linear weights round-tripped through IF4
                    (the FPGA's vision quantization) → mimics the hardware numerics.

    Both are captured in one model load (full-precision forward first, then quantize
    in place and forward again). Returns (refs, meta) with refs["hf"], refs["hostref"].
    """
    hf_model, model_dir = _ensure_hf_model(SCRIPT_DIR, cfg)
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)

    # Preprocess identically to _run_vision_encoder_fpga (pixels are prompt-independent).
    image = Image.open(image_path).convert("RGB").resize(VISION_CANONICAL_SIZE, Image.BICUBIC)
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_prompt], images=[[image]], return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    pixel_position_ids = inputs["image_position_ids"]
    padding = (pixel_position_ids == -1).all(dim=-1).squeeze(0)  # [S] True = padding patch
    n_patches = pixel_values.shape[1]

    vt = hf_model.model.vision_tower

    def _capture():
        """One get_image_features forward; return {A, B, C} (B = pre-pool encoder out)
        plus layer-0 intermediates L0_before/L0_after/L0_end for --layer0."""
        caps = {"layers": []}
        hooks = [
            vt.patch_embedder.register_forward_hook(lambda m, i, o: caps.__setitem__("A", _tensor(o))),
            vt.encoder.register_forward_hook(lambda m, i, o: caps.__setitem__("B", _tensor(o))),
        ]
        try:
            for L in vt.encoder.layers:
                hooks.append(L.register_forward_hook(
                    lambda m, i, o, _c=caps["layers"]: _c.append(_tensor(o))))
            L0 = vt.encoder.layers[0]
            hooks.append(L0.input_layernorm.register_forward_hook(
                lambda m, i, o: caps.__setitem__("L0_before", _tensor(o))))
            hooks.append(L0.self_attn.register_forward_hook(
                lambda m, i, o: caps.__setitem__("L0_after", _tensor(o))))
            hooks.append(L0.register_forward_hook(
                lambda m, i, o: caps.__setitem__("L0_end", _tensor(o))))
            # raw attention-core output = the INPUT to o_proj (pre-hook grabs args[0])
            hooks.append(L0.self_attn.o_proj.register_forward_pre_hook(
                lambda m, args: caps.__setitem__("L0_attn_core", args[0] if args else None)))
        except Exception as e:
            print(f"  [numeric] (layer-0 hooks unavailable: {type(e).__name__}: {e})")
        with torch.no_grad():
            out = hf_model.model.get_image_features(
                pixel_values=pixel_values.to(torch.bfloat16),
                image_position_ids=pixel_position_ids)
        for h in hooks:
            h.remove()
        caps["C"] = getattr(out, "pooler_output", out)
        if _rows(caps.get("B")) != n_patches and caps["layers"] and _rows(caps["layers"][-1]) == n_patches:
            caps["B"] = caps["layers"][-1]
        _add_l0_attention_layout_refs(caps, vt, pixel_position_ids)
        return caps

    print("  [numeric] reference 1/2: full-precision HF vision tower (hf)")
    hf_refs = _capture()

    nq = quantize_vision_tower_(vt, VISION_QUANT_PRECISION)
    print(f"  [numeric] reference 2/2: HF tower + {nq} Linear weights through "
          f"{VISION_QUANT_PRECISION.upper()} (hostref = mimics FPGA quantization)")
    hostref = _capture()

    del hf_model
    return {"hf": hf_refs, "hostref": hostref}, {"padding": padding}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gemma4 E2B vision-encoder numeric check: host oracle vs FPGA (SNR).")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path (default: shipped yosemite.jpg).")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Text prompt (vision output is prompt-independent; affects nothing here).")
    parser.add_argument("--layer0", action="store_true",
                        help="Deep layer-0 probe: compare FPGA vs host at 3 points inside "
                             "encoder layer 0 (before-attn / after-attn / end-of-L0). "
                             "Bakes 3 snapshot DMAs into the layer-0 ISA; forces a bin rebuild.")
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--dev", type=str, default="xdma0")
    parser.add_argument("--cycle", type=float, default=5.62)
    args = parser.parse_args()

    # Force the FPGA vision path (this branch defaults to the HF host-feature
    # fallback via GEMMA4_VISION_FPGA); numeric.py needs the FPGA readbacks (_vis_ckpt).
    os.environ["GEMMA4_VISION_FPGA"] = "1"
    if args.layer0:
        os.environ["GEMMA4_VIS_L0_CKPT"] = "1"
        print("  [numeric] --layer0: layer-0 checkpoint snapshots enabled (forces bin rebuild)")

    set_dma_device(args.dev)
    g4.DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    g4.DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    g4.DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}")

    image_path = args.image or DEFAULT_IMAGE
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    # --- FPGA side: compile (if needed), load, run the vision encoder ---
    ue = Gemma4_UnifiedEngine(local_weights=args.local_weights)
    bin_dir = os.path.join(SCRIPT_DIR, "gemma4_e2b_bin")
    if args.layer0:
        # --layer0 changes the vision ISA + tensor layout → must rebuild the bin.
        for _f in ("programs.bin", "programs.json"):
            try:
                os.remove(os.path.join(bin_dir, _f))
            except FileNotFoundError:
                pass
    if not (os.path.exists(os.path.join(bin_dir, "programs.bin"))
            and os.path.exists(os.path.join(bin_dir, "programs.json"))):
        print("--- Compile: building LM + vision + audio into programs.bin ---")
        ue.compile_instruction_bin(image_path=DEFAULT_IMAGE, audio_path=DEFAULT_AUDIO)
    ue.load_instruction_bin()

    print(f"\n[numeric] running FPGA vision encoder on {image_path} ...")
    ue.set_prefill_seq_vlm(image_path, prompt=args.prompt)
    ckpt = getattr(ue, "_vis_ckpt", None)
    if not ckpt:
        raise SystemExit("FPGA checkpoints missing (self._vis_ckpt not set).")

    # --- Host side: build BOTH references (hostref = IF4/HW-mimicking, hf = full precision) ---
    print(f"\n[numeric] building references ...")
    refs, meta = build_references(ue._cfg, image_path, args.prompt)
    real_rows = ~meta["padding"]  # non-padding patches

    if args.layer0:
        l0 = getattr(ue, "_vis_l0", None)
        if not l0:
            raise SystemExit("layer-0 snapshots missing (self._vis_l0 not set).")
        def _health(name, t, mask):
            x = t.detach().float()
            x = x[mask] if mask is not None and mask.numel() == x.shape[0] else x
            xf = x[torch.isfinite(x)]
            rms = float(torch.sqrt((xf * xf).mean())) if xf.numel() else float('nan')
            print(f"  [numeric] {name:16s} finite={int(torch.isfinite(x).sum())}/{x.numel()} "
                  f"nan={int(torch.isnan(x).sum())} inf={int(torch.isinf(x).sum())} rms={rms:.4g} "
                  f"min={float(xf.min()) if xf.numel() else float('nan'):.4g} "
                  f"max={float(xf.max()) if xf.numel() else float('nan'):.4g}")

        print("\n[numeric] ===== FPGA vs references — LAYER 0 (SNR dB; real patches) =====")
        S = real_rows.numel()
        aS = ((S + 63) // 64) * 64
        nh = l0["q_hm"].shape[0] // aS if "q_hm" in l0 else 0
        hm_mask = torch.zeros(nh, aS, dtype=torch.bool)
        if nh:
            hm_mask[:, :S] = real_rows.unsqueeze(0).expand(nh, S)
            hm_mask = hm_mask.reshape(nh * aS)
        def _report_if(name, ref_key, res_key, ref_pack, mask=real_rows):
            if ref_key in ref_pack and res_key in l0:
                report(name, ref_pack[ref_key], l0[res_key], row_mask=mask)

        print("  FPGA vs HOSTREF (IF4 — mimics hardware):")
        report("L0 before-attn", refs["hostref"]["L0_before"], l0["before_attn"], row_mask=real_rows)
        _report_if("L0 q-proj", "L0_q_proj", "q_proj", refs["hostref"])
        _report_if("L0 k-proj", "L0_k_proj", "k_proj", refs["hostref"])
        _report_if("L0 v-proj", "L0_v_proj", "v_proj", refs["hostref"])
        _report_if("L0 q-pre-rope", "L0_q_pre_rope", "q_pre_rope", refs["hostref"])
        _report_if("L0 k-pre-rope", "L0_k_pre_rope", "k_pre_rope", refs["hostref"])
        _report_if("L0 v-norm", "L0_v_norm", "v_norm", refs["hostref"])
        _report_if("L0 rope-q", "L0_rope_q", "rope_q", refs["hostref"])
        _report_if("L0 rope-k", "L0_rope_k", "rope_k", refs["hostref"])
        _report_if("L0 q-hm", "L0_q_hm", "q_hm", refs["hostref"], hm_mask)
        _report_if("L0 k-hm", "L0_k_hm", "k_hm", refs["hostref"], hm_mask)
        _report_if("L0 v-hm", "L0_v_hm", "v_hm", refs["hostref"], hm_mask)
        _report_if("L0 attn-out-hm", "L0_attn_out_hm", "attn_out_hm", refs["hostref"], hm_mask)
        _report_if("L0 attn-core fpga", "L0_attn_core_fpga", "attn_core", refs["hostref"])
        report("L0 attn-core",   refs["hostref"]["L0_attn_core"], l0["attn_core"], row_mask=real_rows)
        report("L0 after-attn",  refs["hostref"]["L0_after"],  l0["after_attn"],  row_mask=real_rows)
        report("L0 end",         refs["hostref"]["L0_end"],    l0["end_l0"],      row_mask=real_rows)
        print("  FPGA vs HF (full-precision ground truth):")
        report("L0 before-attn", refs["hf"]["L0_before"], l0["before_attn"], row_mask=real_rows)
        _report_if("L0 q-proj", "L0_q_proj", "q_proj", refs["hf"])
        _report_if("L0 k-proj", "L0_k_proj", "k_proj", refs["hf"])
        _report_if("L0 v-proj", "L0_v_proj", "v_proj", refs["hf"])
        _report_if("L0 q-pre-rope", "L0_q_pre_rope", "q_pre_rope", refs["hf"])
        _report_if("L0 k-pre-rope", "L0_k_pre_rope", "k_pre_rope", refs["hf"])
        _report_if("L0 v-norm", "L0_v_norm", "v_norm", refs["hf"])
        _report_if("L0 rope-q", "L0_rope_q", "rope_q", refs["hf"])
        _report_if("L0 rope-k", "L0_rope_k", "rope_k", refs["hf"])
        _report_if("L0 q-hm", "L0_q_hm", "q_hm", refs["hf"], hm_mask)
        _report_if("L0 k-hm", "L0_k_hm", "k_hm", refs["hf"], hm_mask)
        _report_if("L0 v-hm", "L0_v_hm", "v_hm", refs["hf"], hm_mask)
        _report_if("L0 attn-out-hm", "L0_attn_out_hm", "attn_out_hm", refs["hf"], hm_mask)
        _report_if("L0 attn-core fpga", "L0_attn_core_fpga", "attn_core", refs["hf"])
        report("L0 attn-core",   refs["hf"]["L0_attn_core"], l0["attn_core"], row_mask=real_rows)
        report("L0 after-attn",  refs["hf"]["L0_after"],  l0["after_attn"],  row_mask=real_rows)
        report("L0 end",         refs["hf"]["L0_end"],    l0["end_l0"],      row_mask=real_rows)
        print("  roped-Q health (no clean HF ref; checks RoPE didn't blow up):")
        _health("L0 rope-Q FPGA", l0["rope_q"], real_rows)

        # RoPE is a rotation → it MUST preserve each row's L2 norm. Compare the
        # per-row norm of Q before RoPE (q_norm output) vs after RoPE. A ratio far
        # from 1.0 means the FPGA 2D-RoPE is broken (magnitude not preserved).
        if "q_pre_rope" in l0:
            pre = l0["q_pre_rope"].float()[real_rows]
            post = l0["rope_q"].float()[real_rows]
            npre = torch.linalg.vector_norm(pre, dim=-1)
            npost = torch.linalg.vector_norm(post, dim=-1)
            ratio = (npost / (npre + 1e-9))
            print(f"  [numeric] RoPE norm-preservation: |post|/|pre| per-row "
                  f"mean={float(ratio.mean()):.4f} median={float(ratio.median()):.4f} "
                  f"min={float(ratio.min()):.4f} max={float(ratio.max()):.4f}  "
                  f"(RoPE is a rotation → must be ≈1.0; rms pre={float(torch.sqrt((pre*pre).mean())):.4g} post={float(torch.sqrt((post*post).mean())):.4g})")

        # Self-contained single-head (L0,H0) check: recompute the attention on the
        # EXACT Q/K/V the FPGA core read, and SNR-compare against the FPGA's own OUT.
        # No HF, no head re-assembly — isolates one attention run. If this is high,
        # the core is correct in context and the mismatch is upstream (rope/QKV) or
        # in the head-layout comparison; if low, the core itself is wrong here.
        h0 = getattr(ue, "_vis_l0h0", None)
        if h0 is not None:
            import math as _m
            Q = h0["q"].float(); K = h0["k"].float(); V = h0["v"].float()
            OUT = h0["out"].float(); BIAS = h0["bias"].float()
            HD = Q.shape[-1]
            # Core scales Q by 1/sqrt(HD) internally (Q here is the sqrt(HD)-pre-scaled input).
            scores = (Q * (1.0 / _m.sqrt(HD))) @ K.t() + BIAS
            P = torch.softmax(scores, dim=-1)
            ref = P @ V
            print("  self-contained head-0 attention (host recompute on FPGA's own Q/K/V):")
            report("L0H0 attn self", ref, OUT, row_mask=real_rows)
    else:
        print("\n[numeric] ===== FPGA vs references (SNR dB; higher = closer; real patches) =====")
        print("  FPGA vs HOSTREF (HF tower with IF4-quantized weights — mimics hardware numerics):")
        report("A patch_embed", refs["hostref"]["A"], ckpt["A"], row_mask=real_rows)
        report("B encoder_out", refs["hostref"]["B"], ckpt["B"], row_mask=real_rows)
        report("C image_feat",  refs["hostref"]["C"], ckpt["C"])
        print("  FPGA vs HF (full-precision transformers ground truth):")
        report("A patch_embed", refs["hf"]["A"], ckpt["A"], row_mask=real_rows)
        report("B encoder_out", refs["hf"]["B"], ckpt["B"], row_mask=real_rows)
        report("C image_feat",  refs["hf"]["C"], ckpt["C"])

    print("\n[numeric] done.")


if __name__ == "__main__":
    main()
