"""Unit test of the GPT-OSS card program on one engine: shard 0, layer 0 only, positions 0..NPOS-1.
Each segment's output is compared with a torch reference built from the SAME sliced weights
(dequantized from the card's FP4), so the numbers isolate the card's computation (test only)."""
import json
import os
import sys

import torch
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import gptoss_card as G  # noqa: E402
import quant_lib as Q  # noqa: E402

WORK = os.path.expanduser("~/apex_tp/models/gptoss20b")
TP = os.path.join(WORK, "tp10")
plan = json.load(open(os.path.join(TP, "plan.json")))
cfg = plan["config"]
H, HD = cfg["hidden_size"], cfg["head_dim"]
SHARD = int(os.environ.get("SHARD", "0"))
NPOS = int(os.environ.get("NPOS", "4"))
cdir = os.path.join(TP, "card_%02d" % SHARD)

T, qcfg = G.load_base_port("gptoss_test_port")
shard = G.GptOssShard(T, qcfg, sys.argv[1] if len(sys.argv) > 1 else "xdma0", 0, cdir, os.path.join(WORK, "weights"),
                      plan, max_ctx=int(os.environ.get("CTX", "512")), n_layers=1)
with T._quiet():
    shard.prepare()
vocab = plan["cards"][SHARD]["vocab"]
n_seg = shard.compile(parts_n=1, vocab_n=vocab[1] - vocab[0])
print("compiled: %d segments %s, %d expert subroutines" % (n_seg, "".join(shard.seg_kind), len(shard.sub_addr[0])),
      flush=True)

# ---------------- torch reference from the same slice
w = torch.load(os.path.join(cdir, "layer_00.pt"), weights_only=False)


def deq(pair, N, K):
    s, d = pair
    return Q.dequantize_if4(d.numpy().tobytes(), s.view(torch.int16).numpy().tobytes(), N, K, 64).float()


nq = len(w["q_heads"])
Wq, Wk, Wv = deq(w["q"], nq * HD, H), deq(w["k"], HD, H), deq(w["v"], HD, H)
Wo = deq(w["o"], H, nq * HD)
f = lambda t: t.float()  # noqa: E731
cos, nsin, sin = G.yarn_tables(cfg, 512)


def rms(x, g):
    return x * torch.rsqrt((x * x).mean() + cfg["rms_norm_eps"]) * f(g)


def rope(x, p):
    h = HD // 2
    c, s = cos[p, :h].float(), sin[p, :h].float()
    a, b = x[:h], x[h:]
    return torch.cat([a * c - b * s, b * c + a * s])


def experts_ref(xn2, idx, lg):
    wts = torch.softmax(torch.tensor(lg), 0)
    acc = torch.zeros(H)
    for wi, e in zip(wts, idx):
        ex = w["experts"][e]
        n = ex["gu_b"].numel() // 2
        gu = deq(ex["gu"], 2 * n, H)
        g = (gu[:n] @ xn2 + f(ex["gu_b"][:n])).clamp(max=cfg["swiglu_limit"])
        u = (gu[n:] @ xn2 + f(ex["gu_b"][n:])).clamp(-cfg["swiglu_limit"], cfg["swiglu_limit"])
        hid = g * torch.sigmoid(1.702 * g) * (u + 1)
        y = deq(ex["down"], H, n) @ hid
        if ex["down_b"] is not None:
            y = y + f(ex["down_b"])
        acc += wi * y
    return acc


def rd(addr, n):
    return T._read_bf(shard.ue, addr, (n,)).float()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


torch.manual_seed(0)
Kc, Vc = [], []
win = cfg["sliding_window"]
for pos in range(NPOS):
    x = (torch.randn(H) * float(os.environ.get("XSCALE", "0.5"))).to(torch.bfloat16).float()
    shard.set_position(pos, x.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
    us_a = shard.run(shard.seg_start[0])
    share_a = rd(shard.SHARE_A, H)
    # reference attention of this shard's heads (sink, sliding window on layer 0)
    xn = rms(x, w["ln_in"])
    q = Wq @ xn + f(w["q_b"])
    k = rope(Wk @ xn + f(w["k_b"]), pos)
    v = Wv @ xn + f(w["v_b"])
    Kc.append(k); Vc.append(v)
    lo = max(0, pos - win + 1) if w["sliding"] else 0
    Kk, Vv = torch.stack(Kc[lo:]), torch.stack(Vc[lo:])
    outs = []
    for h in range(nq):
        qh = rope(q[h * HD:(h + 1) * HD], pos)
        sc = torch.cat([Kk @ qh / HD ** 0.5, w["sinks"][h:h + 1].float()])
        p = torch.softmax(sc, 0)[:-1]
        outs.append(p @ Vv)
    ref_a = Wo @ torch.cat(outs) + (f(w["o_b"]) if w["o_b"] is not None else 0)
    # R: the residual after attention is X + share (one shard stands in for the sum here)
    shard.ue.dma_write(T.DMA_DEVICE_H2C, shard.PARTS, T._bf(share_a), H * 2)
    us_r = shard.run(shard.seg_start[1])
    logits = rd(shard.LOGITS, 64)[:32]
    x2 = x + share_a
    xn2 = rms(x2, w["ln_post"])
    ref_l = f(w["router_w"]) @ xn2 + f(w["router_b"])
    idx = torch.topk(logits, 4).indices.tolist()
    lg = [float(logits[i]) for i in idx]
    shard.set_experts(0, idx, lg)
    us_e = shard.run(shard.seg_start[2])
    share_e = rd(shard.SHARE_E, H)
    ref_e = experts_ref(xn2, idx, lg)
    ref_idx = torch.topk(ref_l, 4).indices.tolist()
    print("pos %d | A %.0f us rel err %.4f | R %.0f us rel err %.4f, top4 card %s ref %s | E %.0f us rel err %.4f"
          % (pos, us_a, rel(share_a, ref_a), us_r, rel(logits, ref_l), idx, ref_idx, us_e, rel(share_e, ref_e)),
          flush=True)
