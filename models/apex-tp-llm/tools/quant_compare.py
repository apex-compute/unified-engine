"""Host-side comparison of the card's 4-bit weight formats on real Qwen3.8-27B tensors (preparation
work: nothing runs on the cards). Per matrix, the weight SQNR in dB (higher is better) of:
  FP4      E2M1 x per-64 bf16 scale (what the port uses today)
  INT4     -8..7 x per-64 scale
  IF4mix   per 64-block the better of INT4 / FP4 (the card picks by the scale's sign bit)
  TQ4-NF4  16-entry codebook (NF4 levels) x per-64 absmax scale
  TQ4-fit  16-entry codebook fitted to this matrix (Lloyd/k-means on w/absmax) x per-64 scale
All formats use the same bytes per weight, so on the card they cost the same time."""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import quant_lib as Q  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
MODEL = os.path.expanduser("~/apex_exp/models/q38_27b/qwen3.8_27b_bin/Qwen3.8-27B")
index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_open = {}


def get(key):
    f = index[key]
    if f not in _open:
        _open[f] = safe_open(os.path.join(MODEL, f), "pt")
    return _open[f].get_tensor(key)


NF4 = torch.tensor([-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0])


def sqnr(w, r):
    return 10 * np.log10((w.double() ** 2).sum().item() / max(((w.double() - r.double()) ** 2).sum().item(), 1e-30))


def codebook_quant(w, cb):
    """w [N, K] -> reconstruction with per-64 absmax scale (bf16, positive) and nearest codebook entry."""
    N, K = w.shape
    b = w.float().view(N, K // 64, 64)
    s = b.abs().amax(-1, keepdim=True).clamp(min=1e-8).to(torch.bfloat16).float()
    x = (b / s).flatten()
    cbs, _ = torch.sort(cb.to(torch.bfloat16).float())
    mid = (cbs[1:] + cbs[:-1]) / 2
    idx = torch.bucketize(x, mid)
    return (cbs[idx].view_as(b) * s).view(N, K)


def fit_codebook(w, iters=25, sample=2_000_000):
    N, K = w.shape
    b = w.float().view(N, K // 64, 64)
    s = b.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    x = (b / s).flatten()
    g = torch.Generator().manual_seed(0)
    if x.numel() > sample:
        x = x[torch.randint(0, x.numel(), (sample,), generator=g)]
    cb = NF4.clone()                                         # start from NF4, keep +-1 (the block maxima)
    for _ in range(iters):
        cbs, _ = torch.sort(cb)
        idx = torch.bucketize(x, (cbs[1:] + cbs[:-1]) / 2)
        new = torch.stack([x[idx == i].mean() if (idx == i).any() else cbs[i] for i in range(16)])
        new[0], new[-1] = -1.0, 1.0
        cb = new
    return cb


def compare(name, w):
    w = w.to(torch.bfloat16)
    N, K = w.shape
    res = {}
    for label, iv in (("FP4", False), ("INT4", True), ("IF4mix", None)):
        d, sc = Q.quantize_if4(w, 64, int_variant=iv)
        r = Q.dequantize_if4(d, sc, N, K, 64)
        res[label] = sqnr(w, r)
        if label == "IF4mix":
            s = torch.frombuffer(bytearray(sc), dtype=torch.bfloat16)
            res["int4_share"] = 100.0 * (s.float() < 0).float().mean().item()
    res["TQ4-NF4"] = sqnr(w, codebook_quant(w, NF4))
    cb = fit_codebook(w)
    res["TQ4-fit"] = sqnr(w, codebook_quant(w, cb))
    print("%-48s %6d x %-6d | FP4 %5.2f | INT4 %5.2f | IF4mix %5.2f (INT4 blocks %4.1f%%) | TQ4-NF4 %5.2f | TQ4-fit %5.2f"
          % (name, N, K, res["FP4"], res["INT4"], res["IF4mix"], res["int4_share"], res["TQ4-NF4"], res["TQ4-fit"]),
          flush=True)
    return res


layers = [int(v) for v in os.environ.get("LAYERS", "0,3,31,42,62,63").split(",")]
cfg = json.load(open(os.path.join(MODEL, "config.json")))["text_config"]
full = {i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"}
rows = []
t0 = time.time()
for i in layers:
    p = "model.language_model.layers.%d." % i
    names = ((["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] if i in full else
              ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"])
             + ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    for n in names:
        rows.append(compare("L%02d %s" % (i, n), get(p + n + ".weight")))
rows.append(compare("lm_head (rows 0..32767)", get("lm_head.weight")[:32768]))
keys = ["FP4", "INT4", "IF4mix", "TQ4-NF4", "TQ4-fit"]
print("MEAN over %d matrices: " % len(rows) + " | ".join("%s %.2f dB" % (k, np.mean([r[k] for r in rows])) for k in keys))
print("MIN  over %d matrices: " % len(rows) + " | ".join("%s %.2f dB" % (k, np.min([r[k] for r in rows])) for k in keys))
print("done in %.0f s" % (time.time() - t0))
