"""Validate the keystone (items 1+2): SEPARATE q/k/v linears + real relative-position
bias (HF Swin structure) through the generic frontend + IRExecutor, vs torch."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from loom_ir import IRExecutor
from validate_attention import snr_db


class WinAttn(nn.Module):
    """HF-style windowed self-attention: separate q/k/v, additive relpos bias."""
    def __init__(self, dim, nh, ws, mask=None):
        super().__init__()
        self.dim, self.nh, self.ws, self.hd = dim, nh, ws, dim // nh
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        wa = ws * ws
        self.register_buffer("rpb", torch.randn(nh, wa, wa) * 0.1)
        if mask is not None:
            self.register_buffer("mask", mask)         # [nw, wa, wa]
        else:
            self.mask = None

    def forward(self, x):                              # x: [nw, wa, dim]
        nw, wa, dim = x.shape
        def hsplit(t): return t.view(nw, wa, self.nh, self.hd).permute(0, 2, 1, 3)
        q, k, v = hsplit(self.q(x)), hsplit(self.k(x)), hsplit(self.v(x))
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
        a = a + self.rpb.unsqueeze(0)
        if self.mask is not None:
            a = a + self.mask.view(nw, 1, wa, wa)
        a = a.softmax(-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(nw, wa, dim)
        return self.proj(o)


def run(dim=192, nh=6, ws=12, nw=4, shifted=False):
    torch.manual_seed(0)
    wa = ws * ws
    mask = None
    if shifted:
        mask = torch.zeros(nw, wa, wa)
        mask[1:] = (torch.randn(nw - 1, wa, wa) > 1.2).float() * -100.0   # sparse -100 mask
    m = WinAttn(dim, nh, ws, mask=mask).eval()
    x = torch.randn(nw, wa, dim)
    ref = m(x).reshape(nw * wa, dim)
    ex = IRExecutor(m, (x,), ws=ws)
    print(f"  attn spans={len(ex.attn)}  precomputed={list(ex.em.precomputed.keys())}")
    hw, last = ex.run(banner=False)
    s = snr_db(ref, hw[:nw * wa, :dim])
    tag = "SHIFTED(mask)" if shifted else "non-shifted"
    print(f"[real-attn {tag}] SNR = {s:.2f} dB  [{'PASS' if s > 19 else 'FAIL'}]")
    print(f"   ref[0,:4]={[round(v,4) for v in ref[0,:4].tolist()]}")
    print(f"   hw [0,:4]={[round(v,4) for v in hw[0,:4].tolist()]}")
    return s


if __name__ == "__main__":
    print("=== non-shifted (relpos only) ===");  run(shifted=False)
    print("=== shifted (relpos + mask) ===");     run(shifted=True)
