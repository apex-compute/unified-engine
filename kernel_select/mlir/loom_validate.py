#!/usr/bin/env python3
"""loom_validate.py — execute a compiled ue program on the FPGA and validate vs torch.

Proves the whole pipeline end to end NUMERICALLY: torch module -> ue MLIR -> walker
-> captured program -> program_execute on hardware -> read back -> SNR vs torch eager.

Starts on a 64-ALIGNED MLP block (dim=128) so every op is numerically clean (no
pad64 normalization issue, no attention). This validates the execute machinery;
the 96-padding (Swin-Tiny) + attention staging layer on top of this proven harness.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_to_ue
from loom_walker import LoomWalker, parse_mlir, _flat2d, _attrs_for, _EMIT, _ALIAS, _DEFER
from ir_harness import pad64


class MLPBlock(nn.Module):
    """64-aligned transformer MLP sub-block: x + fc2(gelu(fc1(norm(x))))."""
    def __init__(self, dim=128, mlp=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp)
        self.fc2 = nn.Linear(dim * mlp, dim)

    def forward(self, x):                       # x: [M, dim]
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


def _pad_last(t, n):
    """Zero-pad a 1D/2D tensor's last dim up to n."""
    if t.dim() == 1 and t.shape[0] < n:
        return F.pad(t, (0, n - t.shape[0]))
    if t.dim() == 2 and t.shape[1] < n:
        return F.pad(t, (0, n - t.shape[1]))
    return t


class ValidatingWalker(LoomWalker):
    """Walker that loads REAL weights+input, then executes and exposes outputs."""
    def __init__(self, mlir, argvals, *, ue):
        self._argvals = argvals
        super().__init__(mlir, stage_reg="tok", ue=ue)

    def prep(self):
        for (nm, shp), val in zip(self._args, self._argvals):
            t = self.arena.alloc_weight(_flat2d(shp), "bf16", name=nm)
            data = val.reshape(_flat2d(shp)) if val.dim() > 2 else val
            data = _pad_last(data, t.shape[-1])
            self.ue.dma_to_accelerator_memory(t.addr, data.to(torch.bfloat16))
            self.sym[nm] = t
        for o in self._ops:
            self.sym[o["res"]] = self.arena.alloc(_flat2d(o["out"]), name=o["res"])


def snr_db(ref, hw):
    diff = (ref - hw).abs()
    return float(20 * torch.log10(ref.abs().mean() / (diff.mean() + 1e-10)))


def validate_mlp(dim=128, M=64):
    torch.manual_seed(0)
    model = MLPBlock(dim=dim).eval()
    x = torch.randn(M, dim)
    ref = model(x)                              # torch eager reference

    mlir, em, argvals = torch_to_ue.lower(model, (x,), func_name="mlp")
    print("[validate] torch -> ue MLIR:")
    for ln in mlir.splitlines():
        if "ue." in ln:
            print("   ", ln.strip())

    from user_dma_core import UnifiedEngine
    ue = UnifiedEngine()
    w = ValidatingWalker(mlir, argvals, ue=ue)

    last = w._ops[-1]["res"]
    out_t = w.sym[last]
    w.execute("forward", bind={w.gpr("tok"): M}, banner=False)
    hw = ue.dma_from_accelerator_memory(out_t.addr, tuple(out_t.shape)).float()
    hw = hw[:, :dim]                            # drop any pad columns

    s = snr_db(ref, hw)
    print(f"\n[validate] output {tuple(ref.shape)}  SNR = {s:.2f} dB  "
          f"[{'PASS' if s > 19 else 'FAIL'}]")
    print(f"           ref[0,:4] = {ref[0,:4].tolist()}")
    print(f"           hw [0,:4] = {hw[0,:4].tolist()}")
    return s


if __name__ == "__main__":
    validate_mlp()
