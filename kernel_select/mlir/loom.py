#!/usr/bin/env python3
"""loom.py — weave a torch model into unified-engine hardware ops via MLIR.

The MLIR-path analog of k_select.py. Drives the full pipeline:

    HF/torch model
      ├─ 1. export    torch.export()              -> ATen FX graph
      ├─ 2. emit      torch_to_ue.lower()         -> ue MLIR            [WORKS]
      ├─ 3. verify    standalone-opt (round-trip) -> type-checked IR    [WORKS]
      ├─ 4. raise     standalone-opt --raise-ue   -> fused window/head  [STUB]
      └─ 5. run       translate -> dispatch/Arena -> program_execute    [STUB]

Run:
    python loom.py                 # built-in Swin block demo (offline)
    python loom.py --model microsoft/swin-tiny-patch4-window7-224
    python loom.py --emit-only     # stop after step 3, print the IR
"""
import os
import sys
import argparse
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.join(HERE, "llvm-project", "build", "bin", "standalone-opt")

import torch_to_ue


# =============================================================================
# Model loading — demo block (offline) or a real HF model.
# =============================================================================
def load_demo():
    """Built-in Swin W-MSA block; runs with no downloads."""
    m = torch_to_ue.SwinBlock().eval()
    return m, (torch.randn(1, 56, 56, 96),), "swin_block"


def load_hf(model_id):
    """Load a real HF model + a representative example input."""
    from transformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    # vision encoder: image tensor; default to 224x224x3.
    img = getattr(cfg, "image_size", 224)
    ch = getattr(cfg, "num_channels", 3)
    ex = (torch.randn(1, ch, img, img),)
    name = model_id.split("/")[-1].replace("-", "_").replace(".", "_")
    return model, ex, name


# =============================================================================
# Pipeline stages
# =============================================================================
def stage_emit(model, example_inputs, name, strict=True, patch_spec=None):
    """Steps 1+2: torch.export -> ue MLIR text. strict=True raises UnsupportedOp
    on any op with no ue-HLO kernel; strict=False emits poisoned markers.
    patch_spec (from Layer A's plan.patch) names the patch-embed conv by role."""
    mlir, em, _ = torch_to_ue.lower(model, example_inputs, func_name=name,
                                    strict=strict, patch_spec=patch_spec)
    unhandled = [f for f in em.folded if f.startswith("UNHANDLED")]
    return mlir, unhandled


def _patch_spec_for(model_id):
    """Ask Layer A for the declared patch embedding (in_ch/out_dim/patch), if any."""
    if not model_id:
        return None
    try:
        import loom_plan
        return loom_plan.plan_for(model_id).patch or None
    except Exception:
        return None


def stage_verify(mlir_text, name):
    """Step 3: round-trip through standalone-opt; raises on a type/verify error."""
    if not os.path.exists(OPT):
        raise FileNotFoundError(f"standalone-opt not built at {OPT}")
    p = subprocess.run([OPT], input=mlir_text, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"MLIR verify FAILED:\n{p.stderr}")
    return p.stdout


def stage_raise(mlir_text):
    """Step 4: ue->ue fusion — raise reshape+permute+reshape idioms into
    ue.window_partition/window_reverse and head-split. Guarded on EXACT shape
    signatures (a generic reshape+permute+reshape is NOT necessarily a window).
    """
    raise NotImplementedError(
        "stage_raise: --raise-ue fusion pass not built yet (next step). "
        "Generic reshape/permute lower correctly as a fallback.")


def stage_partition(mlir_text):
    """Step 4b: host/device split — classify every op as DEVICE (ue kernel),
    MACRO (attention cluster), VIEW (free reshape/select), or HOST (CPU/torch)."""
    import loom_walker
    import loom_exec
    _, ops = loom_walker.parse_mlir(mlir_text)
    roles = loom_exec.classify(ops)
    return loom_exec.partition_report(ops, roles)


def stage_run(is_demo):
    """Step 5: EXECUTE on the FPGA. Runs a FULL Swin model — HOST patch-embed ->
    DEVICE encoder (stages + patch-merges, the heavy transformer) -> HOST head
    (norm/pool/classifier) -> logits. Reports the predicted class vs torch."""
    import loom_exec
    if not is_demo:
        raise NotImplementedError(
            "stage_run: end-to-end execution wired for the Swin demo model. "
            "Arbitrary --model needs per-model weight loading — emit/verify work today.")
    stages = [(192, 6, 1, 24), (384, 12, 1, 12)]     # 2 stages + a patch-merge
    lref, lhw, hsnr = loom_exec.run_swin_model(stages)
    top_ref, top_hw = int(lref.argmax()), int(lhw.argmax())
    match = "MATCH" if top_ref == top_hw else "MISMATCH"
    print(f"[loom]      hidden-state SNR (device vs torch) = {hsnr:.2f} dB")
    print(f"[loom]      predicted class: torch={top_ref}  fpga={top_hw}  [{match}]")
    return hsnr


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="default: built-in Swin block demo (no downloads).")
    ap.add_argument("--model", default=None, help="HF model id (else demo block)")
    ap.add_argument("--emit-only", action="store_true", help="stop after verify; print IR")
    ap.add_argument("--allow-unsupported", action="store_true",
                    help="emit poisoned ue.unsupported markers instead of failing on missing kernels")
    ap.add_argument("-o", "--out", default=None, help="write verified IR to this .mlir")
    args = ap.parse_args()

    print(f"[loom] loading {'demo Swin block' if not args.model else args.model}")
    model, ex, name = load_demo() if not args.model else load_hf(args.model)

    print("[loom] 1+2  export + emit  (torch.export -> ue MLIR)")
    try:
        patch_spec = _patch_spec_for(args.model)
        if patch_spec:
            print(f"[loom]      patch-embed (Layer A): in_ch={patch_spec.get('in_ch')} "
                  f"out_dim={patch_spec.get('out_dim')} patch={patch_spec.get('patch')}")
        mlir, unhandled = stage_emit(model, ex, name,
                                     strict=not args.allow_unsupported, patch_spec=patch_spec)
    except torch_to_ue.UnsupportedOp as e:
        print(f"[loom] STOP — model needs {len(e.gaps)} op(s) with no ue-HLO kernel:")
        from collections import Counter
        for tgt, cnt in Counter(t for t, _ in e.gaps).most_common():
            print(f"[loom]        {cnt:3d}x  {tgt}")
        print("[loom] add a rule in torch_to_ue.handle + a KernelSpec in hlo.py, "
              "or re-run with --allow-unsupported to emit poisoned markers.")
        return 2
    if unhandled:
        print(f"[loom]      WARNING {len(unhandled)} unhandled ATen ops:")
        for u in sorted(set(unhandled)):
            print(f"[loom]        {u}")

    print("[loom] 3    verify  (standalone-opt round-trip)")
    verified = stage_verify(mlir, name)
    print("[loom]      OK — IR type-checks")

    if args.out:
        with open(args.out, "w") as f:
            f.write(verified)
        print(f"[loom]      wrote {args.out}")

    if args.emit_only:
        print("\n" + verified)
        return 0

    print("[loom] 4    raise  (window/head fusion)")
    try:
        verified = stage_raise(verified)
    except NotImplementedError as e:
        print(f"[loom]      SKIP — {e}")

    print("[loom] 4b   partition  (host/device split)")
    stage_partition(verified)

    print("[loom] 5    run  (execute on FPGA -> read back hidden state)")
    try:
        stage_run(is_demo=args.model is None)
    except NotImplementedError as e:
        print(f"[loom]      SKIP — {e}")

    print("[loom] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
