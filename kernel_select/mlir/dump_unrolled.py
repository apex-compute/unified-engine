#!/usr/bin/env python3
"""dump_unrolled.py — emit the FULLY-UNROLLED ue IR for a Swin-Tiny-shaped stack.

The point: show what the loop reroller will consume. torch.export gives a flat,
unrolled op list — every block of every stage spelled out back-to-back. We stack
real SwinBlocks at Swin-Tiny's depths/dims so the repetition is faithful, then
print the IR with per-block markers so the periodicity is visible by eye.
"""
import torch
import torch.nn as nn
import torch_to_ue
from torch_to_ue import SwinBlock

# Swin-LARGE (the one in models/swin, swin-large-patch4-window12-384-in22k):
#   embed_dim 192, depths [2,2,18,2], heads [6,12,24,48], window 12,
#   dims [192,384,768,1536], spatial [96,48,24,12].
WS = 12
STAGES = [
    dict(dim=192,  heads=6,  depth=2,  res=96),
    dict(dim=384,  heads=12, depth=2,  res=48),
    dict(dim=768,  heads=24, depth=18, res=24),
    dict(dim=1536, heads=48, depth=2,  res=12),
]


class Stage(nn.Module):
    """One Swin stage: `depth` identical W-MSA blocks at fixed dim (no merge)."""
    def __init__(self, dim, heads, depth, ws=WS):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SwinBlock(dim=dim, heads=heads, ws=ws) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def dump_stage(s):
    m = Stage(s["dim"], s["heads"], s["depth"]).eval()
    x = torch.randn(1, s["res"], s["res"], s["dim"])
    mlir, em, _ = torch_to_ue.lower(m, (x,), func_name=f"stage_d{s['dim']}")
    body = [ln for ln in mlir.splitlines() if "ue." in ln]
    print(f"\n{'='*78}\nSTAGE dim={s['dim']} heads={s['heads']} depth={s['depth']} "
          f"res={s['res']}  ->  {len(body)} ops total\n{'='*78}")
    # find per-block period: ops / depth
    per = len(body) // s["depth"]
    for bi in range(s["depth"]):
        print(f"--- block {bi}  ({per} ops) ---")
        for ln in body[bi*per:(bi+1)*per]:
            # strip the long type signature for readability
            print("   ", ln.strip().split(" : ")[0])
    return body, per


if __name__ == "__main__":
    print("FULLY-UNROLLED Swin-LARGE ue IR  (depths [2,2,18,2], dims [192,384,768,1536], window 12)")
    print("This is exactly the flat op list the loop reroller will segment.\n")
    for s in STAGES:
        dump_stage(s)
