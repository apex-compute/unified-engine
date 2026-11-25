"""Prepare the vision tower of a Qwen3.5-family model for the cards (host-side preparation only).

The card kernels need every matrix dimension to be a multiple of 64. The 27B vision tower has
16 heads of 72 dims and an MLP width of 4304, so the weights are laid out padded:

  head lanes   each head gets 128 lanes: its first 36 dims in lanes 0..35, its last 36 in lanes
               64..99, zeros elsewhere. Rotary position encoding pairs dim i with i+36, which in
               this layout is lane i with lane i+64, exactly what the card's rotate-half does for
               a 128-wide head. Q, K and V rows of the QKV projection are moved to these lanes and
               the output projection's columns follow them.
  MLP          fc1 rows / fc2 columns padded with zeros to the next multiple of 64.

Zero rows and columns are exact in the 4-bit format, so the padded tower computes the same
function. Output: $APEX_TP_WORK/models/<model>/vision/vision_weights.pt

    Q_MODEL=q38_27b python vision_prep.py
"""
import glob
import importlib.util
import json
import os
import sys
import time

import torch
from safetensors import safe_open

BASE = os.path.join(os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp")), "models", os.environ.get("Q_MODEL", "q38_27b"))
MODEL = os.path.dirname(sorted(glob.glob(os.path.join(BASE, "*_bin", "*", "config.json")))[0])
OUT = os.path.join(BASE, "vision")
os.makedirs(OUT, exist_ok=True)

spec = importlib.util.spec_from_file_location("vision_prep_port", os.path.join(BASE, "qwen3.5_2b_test.py"))
T = importlib.util.module_from_spec(spec)
sys.modules["vision_prep_port"] = T
spec.loader.exec_module(T)

vc = json.load(open(os.path.join(MODEL, "config.json")))["vision_config"]
HID, HEADS, DEPTH, MLP = vc["hidden_size"], vc["num_heads"], vc["depth"], vc["intermediate_size"]
HD = HID // HEADS
HALF = HD // 2
HDP = 128                                   # padded head width
assert HD <= HDP and HALF <= 64 and HID % 64 == 0
QD = HEADS * HDP
MLPP = -(-MLP // 64) * 64
K_PIXEL = vc["in_channels"] * vc["temporal_patch_size"] * vc["patch_size"] ** 2
MERGE = vc["spatial_merge_size"] ** 2
K_FC = HID * MERGE
OUT_H = vc["out_hidden_size"]
assert K_PIXEL % 64 == 0 and K_FC % 64 == 0 and OUT_H % 64 == 0

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_open = {}


def get(key):
    f = index["model.visual." + key]
    if f not in _open:
        _open[f] = safe_open(os.path.join(MODEL, f), "pt")
    return _open[f].get_tensor("model.visual." + key).float()


# lane of head dim j inside a padded head
lane = torch.tensor([j if j < HALF else 64 + (j - HALF) for j in range(HD)])
head_lanes = torch.cat([h * HDP + lane for h in range(HEADS)])              # [HID] -> position in [QD]
qkv_rows = torch.cat([p * QD + head_lanes for p in range(3)])               # [3*HID] -> position in [3*QD]

bf = lambda t: t.to(torch.bfloat16).contiguous()
t0 = time.time()
W = {"dims": {"HID": HID, "HEADS": HEADS, "HD": HD, "HDP": HDP, "QD": QD, "MLP": MLP, "MLPP": MLPP, "DEPTH": DEPTH,
              "K_PIXEL": K_PIXEL, "K_FC": K_FC, "OUT": OUT_H, "MERGE": MERGE, "patch": vc["patch_size"],
              "tps": vc["temporal_patch_size"], "grid": int(vc["num_position_embeddings"] ** 0.5), "theta": 10000.0}}
W["patch_w"] = bf(get("patch_embed.proj.weight").reshape(HID, K_PIXEL))
W["patch_b"] = bf(get("patch_embed.proj.bias"))
W["pos_table"] = bf(get("pos_embed.weight"))
layers = []
for li in range(DEPTH):
    p = "blocks.%d." % li
    qkv_w, qkv_b = get(p + "attn.qkv.weight"), get(p + "attn.qkv.bias")
    assert qkv_w.shape == (3 * HID, HID), qkv_w.shape
    qw = torch.zeros(3 * QD, HID); qw[qkv_rows] = qkv_w
    qb = torch.zeros(3 * QD); qb[qkv_rows] = qkv_b
    proj_w = get(p + "attn.proj.weight")
    pw = torch.zeros(HID, QD); pw[:, head_lanes] = proj_w
    fc1_w, fc1_b = get(p + "mlp.linear_fc1.weight"), get(p + "mlp.linear_fc1.bias")
    fc2_w = get(p + "mlp.linear_fc2.weight")
    assert fc1_w.shape == (MLP, HID) and fc2_w.shape == (HID, MLP)
    f1 = torch.zeros(MLPP, HID); f1[:MLP] = fc1_w
    f1b = torch.zeros(MLPP); f1b[:MLP] = fc1_b
    f2 = torch.zeros(HID, MLPP); f2[:, :MLP] = fc2_w
    layers.append({
        "ln1_w": bf(get(p + "norm1.weight")), "ln1_b": bf(get(p + "norm1.bias")),
        "ln2_w": bf(get(p + "norm2.weight")), "ln2_b": bf(get(p + "norm2.bias")),
        "qkv_b": bf(qb), "proj_b": bf(get(p + "attn.proj.bias")), "fc1_b": bf(f1b), "fc2_b": bf(get(p + "mlp.linear_fc2.bias")),
        "qkv": T.quantize_fp4_64(qw), "proj": T.quantize_fp4_64(pw), "fc1": T.quantize_fp4_64(f1), "fc2": T.quantize_fp4_64(f2)})
    print("LAYER|%d|%.1f s" % (li, time.time() - t0), flush=True)
W["layers"] = layers
mn = get("merger.norm.weight")
assert mn.shape == (HID,), "merger norm is applied per patch in this port (got %s)" % (tuple(mn.shape),)
m1, m2 = get("merger.linear_fc1.weight"), get("merger.linear_fc2.weight")
assert m1.shape == (K_FC, K_FC) and m2.shape == (OUT_H, K_FC), (m1.shape, m2.shape)
W["merger"] = {"ln_w": bf(mn), "ln_b": bf(get("merger.norm.bias")), "fc1_b": bf(get("merger.linear_fc1.bias")),
               "fc2_b": bf(get("merger.linear_fc2.bias")), "fc1": T.quantize_fp4_64(m1), "fc2": T.quantize_fp4_64(m2)}
dst = os.path.join(OUT, "vision_weights.pt")
torch.save(W, dst + ".tmp")
os.replace(dst + ".tmp", dst)
print("DONE|%.0f s|%.0f MB|%s" % (time.time() - t0, os.path.getsize(dst) / 1e6, W["dims"]), flush=True)
