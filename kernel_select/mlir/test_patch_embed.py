#!/usr/bin/env python3
"""test_patch_embed.py — verify ROLE-BASED patch-embed detection.

A conv becomes ue.patching ONLY if it matches the Layer-A declared patch_spec.
Every other conv (overlapping / padded / wrong dims / not-the-patch-embed) fails
closed as UnsupportedOp — never silently miscompiled into patching.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch.nn as nn

import torch_to_ue
from torch_to_ue import lower, UnsupportedOp

OPT = os.path.join(HERE, "llvm-project", "build", "bin", "standalone-opt")


class PatchEmbed(nn.Module):
    """HF-style patch embedding: conv(kernel=stride=patch) -> flatten -> transpose."""
    def __init__(self, in_ch=3, embed=192, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, kernel_size=patch, stride=patch)

    def forward(self, x):                       # x: [B, C, H, W]
        return self.proj(x).flatten(2).transpose(1, 2)   # -> [B, tokens, embed]


class RealConvNet(nn.Module):
    """A genuine online conv (overlapping 3x3 stride-1 pad-1) — NOT patchify."""
    def __init__(self, in_ch=3, out=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


def run():
    ok = True
    SPEC = {"in_ch": 3, "out_dim": 192, "patch": 4}

    # ---- case 1: matching patch-embed -> ue.patching, tail absorbed ----
    print("== case 1: declared patch-embed (3->192, patch4) ==")
    m = PatchEmbed(3, 192, 4).eval()
    mlir, em, _ = lower(m, (torch.randn(1, 3, 224, 224),), func_name="pe", patch_spec=SPEC)
    body = [l.strip() for l in mlir.splitlines() if "ue." in l]
    for l in body:
        print("   ", l.split(" : ")[0])
    has_patching = any("ue.patching" in l for l in body)
    has_permute = any("ue.permute" in l for l in body)
    pe_folds = [f for f in em.folded if f.startswith("PE_FOLD")]
    print(f"   -> ue.patching={has_patching}  ue.permute={has_permute}  "
          f"tail-folded={pe_folds}  unsupported={em.unsupported}")
    # tokens = (224/4)^2 = 3136, embed 192
    has_shape = any("3136x192" in l for l in body if "patching" in l)
    if not (has_patching and not has_permute and pe_folds and not em.unsupported and has_shape):
        print("   FAIL"); ok = False
    else:
        print("   PASS — conv->patching, flatten/transpose absorbed, no permute")

    # ---- case 2: real overlapping conv -> fail closed ----
    print("\n== case 2: real online conv (3x3 s1 p1) ==")
    try:
        lower(RealConvNet().eval(), (torch.randn(1, 3, 32, 32),), patch_spec=SPEC)
        print("   FAIL — should have raised UnsupportedOp"); ok = False
    except UnsupportedOp as e:
        tgts = sorted({g[0] for g in e.gaps})
        print(f"   raised UnsupportedOp, gaps={tgts}")
        print("   PASS — real conv fails closed (not turned into patching)")

    # ---- case 3: patchify conv that ISN'T the declared patch-embed -> fail closed ----
    print("\n== case 3: patchify conv but wrong out_dim (3->64, patch4) ==")
    try:
        lower(PatchEmbed(3, 64, 4).eval(), (torch.randn(1, 3, 32, 32),), patch_spec=SPEC)
        print("   FAIL — should have raised (out_dim 64 != declared 192)"); ok = False
    except UnsupportedOp:
        print("   PASS — non-matching patchify NOT mislabeled as patch-embed")

    # ---- case 4: no patch_spec at all -> conv fails closed ----
    print("\n== case 4: no patch_spec declared ==")
    try:
        lower(PatchEmbed(3, 192, 4).eval(), (torch.randn(1, 3, 32, 32),))
        print("   FAIL — should have raised (no declared patch-embed)"); ok = False
    except UnsupportedOp:
        print("   PASS — without a declared role, a conv is never assumed patch-embed")

    # ---- case 5: emitted IR round-trips through standalone-opt (ue.patching parses) ----
    print("\n== case 5: standalone-opt verify of the patch-embed IR ==")
    if os.path.exists(OPT):
        p = subprocess.run([OPT], input=mlir, capture_output=True, text=True)
        if p.returncode == 0:
            print("   PASS — ue.patching type-checks in the dialect")
        else:
            print(f"   FAIL — standalone-opt rejected:\n{p.stderr.strip()}"); ok = False
    else:
        print(f"   SKIP — standalone-opt not built at {OPT}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
