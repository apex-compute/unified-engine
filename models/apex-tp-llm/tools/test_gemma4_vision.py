"""Unit test of the Gemma 4 image encoder card program on one engine.

Compares the card with a torch emulation built from the SAME vision.pt (IF8 dequantized, fp32 math), so the
numbers isolate the card's computation: first X after DEPTH_TEST layers, then the full soft tokens.
Test only: the reference never runs during inference.
    usage: test_gemma4_vision.py xdmaN [engine] [image]       env: SOFT (70), DEPTH_TEST (1)
"""
import os
import sys
import time

import torch
torch.set_num_threads(2)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4"))
sys.path.insert(0, os.path.expanduser("~/unified-engine"))
import gemma4_card as G  # noqa: E402
import gemma4_vision as GV  # noqa: E402
import quant_lib as Q  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "xdma4"
ENGINE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
IMAGE = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test_samples", "people.jpg")
SOFT = int(os.environ.get("SOFT", "70"))
DEPTH_TEST = [int(v) for v in os.environ.get("DEPTH_TEST", "1").split(",")]
VPATH = os.path.expanduser("~/apex_tp/models/gemma4_26b/vision/vision.pt")
MODEL = os.path.expanduser("~/apex_exp/models/gemma4_26b/hf/gemma-4-26b-a4b-it")

from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # noqa: E402
from PIL import Image  # noqa: E402
inp = Gemma4ImageProcessor(max_soft_tokens=SOFT)(images=[Image.open(IMAGE).convert("RGB")], return_tensors="pt")
pv, pid = inp["pixel_values"], inp["image_position_ids"]

# ---------------- card
T, qcfg = G.load_base_port("gemma4_vision_test_port")
U = T.user_dma_core
T.set_dma_device(DEV)
T.DMA_DEVICE_H2C, T.DMA_DEVICE_C2H = U.DMA_DEVICE_H2C, U.DMA_DEVICE_C2H
U.configure_clock_from_hardware()
ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=qcfg, BASE_ADDR=U.UE_0_BASE_ADDR + ENGINE * 0x10000)
ue.software_reset()
ue._preallocate_identity_matrix()
vc = GV.VisionCard(T, ue, VPATH)
with T._quiet():
    print("weights uploaded in %.0f s" % vc.load(), flush=True)

# ---------------- torch emulation from the same vision.pt (test only)
V = torch.load(VPATH, weights_only=False)
D = V["dims"]
HID, NH, HDP, POS = D["HID"], D["NH"], D["HDP"], D["POS"]


def deq(t, N, K):
    s, d = t
    return Q.dequantize_if8(d.numpy().tobytes(), s.view(torch.int16).numpy().tobytes(), N, K, 64).float()


def rms(x, g=None):
    y = x * (x.pow(2).mean(-1, keepdim=True) + D["EPS"]).pow(-0.5)
    return y if g is None else y * g.float()


px, pp, n_soft, grid = GV.plan_patches(pv, pid, D["POOL"])
L_ = px.shape[0]


def emulate(depth, start=0, x0=None):
    oh = torch.zeros(L_, 2 * POS); oh[torch.arange(L_), pp[:, 0]] = 1; oh[torch.arange(L_), POS + pp[:, 1]] = 1
    x = (2 * px - 1) @ V["patch_w"].float().t() + oh @ V["pos_t"].float().t()
    if start:
        x = x0.clone()
    cs = oh @ V["rope_t"].float().t() + V["rope_bias"].float()
    cos, sinS = cs[:, :HDP], cs[:, HDP:]
    swap = lambda h: torch.cat([h[..., 64:], h[..., :64]], -1)  # noqa: E731
    for w in V["layers"][start:depth]:
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
        return x
    pool = x.view(9, n_soft, HID).sum(0)
    return rms(pool @ V["std_diag"].float() + V["std_c"].float()) @ V["embed_w"].float().t()


def compare(name, card, ref):
    card = card.float()
    rel = (card - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)
    cos = torch.nn.functional.cosine_similarity(card, ref, dim=-1)
    print("%s: rows %d | rel err mean %.4f max %.4f | cosine mean %.5f min %.5f | finite %s" % (
        name, card.shape[0], float(rel.mean()), float(rel.max()), float(cos.mean()), float(cos.min()),
        bool(torch.isfinite(card).all())), flush=True)


print("image %s: %d patches (grid %dx%d) -> %d soft tokens" % (os.path.basename(IMAGE), L_, grid[0], grid[1], n_soft),
      flush=True)
if os.environ.get("PER_LAYER"):                 # one layer at a time from the emulation's exact input
    for li in [int(v) for v in os.environ["PER_LAYER"].split(",")]:
        x_li = emulate(li) if li else None
        if li == 0:
            with T._quiet():
                out, info = vc.encode(pv, pid, depth=1)
            ref = emulate(1)
        else:
            with T._quiet():
                out, info = vc.encode(pv, pid, depth=li + 1, start=li, x_in=x_li)
            ref = emulate(li + 1, li, x_li)
        compare("layer %d alone" % li, out, ref)
        r = ((out.float() - ref).norm(dim=-1) / ref.norm(dim=-1))
        top = torch.topk(r, 3).indices.tolist()
        print("   worst rows %s pos %s |x| %s" % (top, [tuple(pp[i].tolist()) for i in top],
                                             [round(float(ref[i].norm()), 1) for i in top]), flush=True)
    sys.exit(0)
for depth in DEPTH_TEST + ([] if os.environ.get("NO_FULL") else [D["DEPTH"]]):
    with T._quiet():
        out, info = vc.encode(pv, pid, depth=depth)
    print("depth %d: %s" % (depth, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}),
          flush=True)
    t0 = time.time()
    ref = emulate(depth)
    compare("card vs emulation after %d layer(s)" % depth if depth < D["DEPTH"] else "card vs emulation soft tokens",
            out, ref)
torch.save(out, "/tmp/gemma4_vision_card_out.pt")
