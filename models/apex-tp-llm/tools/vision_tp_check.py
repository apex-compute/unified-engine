"""Test only (host): compare the tensor-parallel encoder's output (saved by gemma4_tp.py --image) with a torch
emulation from the same vision.pt (IF8 dequantized, fp32 math), as test_gemma4_vision.py does for one engine.
    vision_tp_check.py /tmp/gemma4_vision_tp_x.pt          X after GEMMA4_VISION_DEPTH layers
    vision_tp_check.py /tmp/gemma4_vision_tp_out.pt IMAGE S   the soft tokens
"""
import os
import sys

import torch
torch.set_num_threads(2)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4"))
sys.path.insert(1, os.path.expanduser("~/unified-engine"))
import gemma4_vision as GV1  # noqa: E402
import quant_lib as Q  # noqa: E402
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # noqa: E402
from PIL import Image  # noqa: E402

V = torch.load(os.path.expanduser("~/apex_tp/models/gemma4_26b/vision/vision.pt"), weights_only=False)
D = V["dims"]
HID, NH, HDP, POS = D["HID"], D["NH"], D["HDP"], D["POS"]
saved = torch.load(sys.argv[1], weights_only=False)
if isinstance(saved, dict):
    card, S, depth, image = saved["x"], saved["S"], saved["depth"], saved["image"]
else:
    card, image, S, depth = saved, sys.argv[2], int(sys.argv[3]), D["DEPTH"]
inp = Gemma4ImageProcessor(max_soft_tokens=S)(images=[Image.open(image).convert("RGB")], return_tensors="pt")
px, pp, n_soft, grid = GV1.plan_patches(inp["pixel_values"], inp["image_position_ids"], D["POOL"])
L_ = px.shape[0]


def deq(t, N, K):
    s, d = t
    return Q.dequantize_if8(d.numpy().tobytes(), s.view(torch.int16).numpy().tobytes(), N, K, 64).float()


def rms(x, g=None):
    y = x * (x.pow(2).mean(-1, keepdim=True) + D["EPS"]).pow(-0.5)
    return y if g is None else y * g.float()


oh = torch.zeros(L_, 2 * POS); oh[torch.arange(L_), pp[:, 0]] = 1; oh[torch.arange(L_), POS + pp[:, 1]] = 1
x = (2 * px - 1) @ V["patch_w"].float().t() + oh @ V["pos_t"].float().t()
cs = oh @ V["rope_t"].float().t() + V["rope_bias"].float()
cos, sinS = cs[:, :HDP], cs[:, HDP:]
swap = lambda h: torch.cat([h[..., 64:], h[..., :64]], -1)  # noqa: E731
for w in V["layers"][:depth]:
    h = rms(x, w["ln_in"])
    q, k, v = [(h @ deq(w[n], NH * HDP, HID).t()).view(L_, NH, HDP) for n in "qkv"]
    q, k, v = rms(q, w["q_norm"]), rms(k, w["k_norm"]), rms(v, V["v_norm"])
    q = q * cos[:, None] + swap(q) * sinS[:, None]
    k = k * cos[:, None] + swap(k) * sinS[:, None]
    a = torch.softmax(torch.einsum("qhd,khd->hqk", q, k), -1)
    o = torch.einsum("hqk,khd->qhd", a, v).reshape(L_, NH * HDP) @ deq(w["o"], HID, NH * HDP).t()
    x = x + rms(o, w["ln_post_attn"])
    h = rms(x, w["ln_pre_ff"])
    g = torch.nn.functional.gelu(h @ deq(w["gate"], D["IP"], HID).t(), approximate="tanh")
    x = x + rms((g * (h @ deq(w["up"], D["IP"], HID).t())) @ deq(w["down"], HID, D["IP"]).t(), w["ln_post_ff"])
if depth < D["DEPTH"]:
    i = torch.arange(L_)
    rows = (i // n_soft) * S + i % n_soft             # the level's fixed layout
    card, ref = card.float()[rows], x
    name = "X after %d layer(s)" % depth
else:
    pool = x.view(9, n_soft, HID).sum(0)
    ref = rms(pool @ V["std_diag"].float() + V["std_c"].float()) @ V["embed_w"].float().t()
    card, name = card.float(), "soft tokens"
rel = (card - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)
cosv = torch.nn.functional.cosine_similarity(card, ref, dim=-1)
print("TP card vs emulation, %s (level %d, %d patches, %d soft tokens): rel err mean %.4f max %.4f | cosine mean "
      "%.5f min %.5f | finite %s" % (name, S, L_, n_soft, float(rel.mean()), float(rel.max()), float(cosv.mean()),
                                    float(cosv.min()), bool(torch.isfinite(card).all())), flush=True)
