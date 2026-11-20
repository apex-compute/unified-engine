"""Unit test of the Gemma 4 card program on one engine: one shard, layers 0 (sliding) and 5 (global),
positions 0..NPOS-1. Each segment's output is compared with a torch reference built from the SAME
sliced weights (dequantized from the card's FP4), so the numbers isolate the card's computation.
Test only: the reference never runs during inference."""
import json
import os
import sys

import torch
torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4"))
sys.path.insert(0, os.path.expanduser("~/unified-engine"))
import gemma4_card as G  # noqa: E402
import quant_lib as Q  # noqa: E402

WORK = os.path.expanduser("~/apex_tp/models/gemma4_26b")
TP = os.path.join(WORK, "tp10")
plan = json.load(open(os.path.join(TP, "plan.json")))
cfg = plan["config"]
H, EPS = cfg["hidden_size"], cfg["rms_norm_eps"]
SHARD = int(os.environ.get("SHARD", "0"))
NPOS = int(os.environ.get("NPOS", "4"))
LAYERS = [int(v) for v in os.environ.get("LAYERS", "0,5").split(",")]
SAMPLE = int(os.environ.get("SAMPLE", "1"))       # full FFN / head reference only every SAMPLE positions (+ the last 3)
cdir = os.path.join(TP, "card_%02d" % SHARD)

T, qcfg = G.load_base_port("gemma4_test_port")
shard = G.Gemma4Shard(T, qcfg, sys.argv[1] if len(sys.argv) > 1 else "xdma0", 0, cdir, os.path.join(WORK, "weights"),
                      plan, max_ctx=int(os.environ.get("CTX", "512")), layers=LAYERS)
with T._quiet():
    shard.prepare()
vocab = plan["cards"][SHARD]["vocab"]
n_seg = shard.compile(parts_n=1, vocab_n=vocab[1] - vocab[0])
print("compiled: %d segments %s, %d expert subroutines per layer" % (n_seg, "".join(shard.seg_kind),
                                                                     len(shard.sub_addr[0])), flush=True)


# ---------------- torch reference from the same slice
def deq(pair, N, K):
    s, d = pair
    return Q.dequantize_if4(d.numpy().tobytes(), s.view(torch.int16).numpy().tobytes(), N, K, 64).float()


f = lambda t: t.float()  # noqa: E731


def rms(x, g=None):
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS)
    return y * f(g) if g is not None else y


def gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def rope_hf(x, pos, kind):
    """transformers' apply_rotary_pos_emb with the default (sliding) or proportional (global) inv_freq."""
    hd = x.shape[-1]
    key = "sliding_attention" if kind == "s" else "full_attention"
    rp = cfg["rope_parameters"][key]
    n = int(rp.get("partial_rotary_factor", 1.0) * hd // 2)
    inv = torch.zeros(hd // 2)
    inv[:n] = 1.0 / (rp["rope_theta"] ** (torch.arange(0, 2 * n, 2, dtype=torch.float32) / hd))
    fr = pos * inv
    emb = torch.cat([fr, fr])
    cos, sin = emb.cos().to(torch.bfloat16).float(), emb.sin().to(torch.bfloat16).float()
    rot = torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1)
    return x * cos + rot * sin


W = {li: torch.load(os.path.join(cdir, "layer_%02d.pt" % li), weights_only=False) for li in LAYERS}
REF = {}
for li, w in W.items():
    hd, nq = w["head_dim"], len(w["q_heads"])
    r = {"hd": hd, "nq": nq}
    if nq:
        r["q"], r["k"], r["o"] = deq(w["q"], nq * hd, H), deq(w["k"], hd, H), deq(w["o"], H, nq * hd)
        r["v"] = deq(w["v"], hd, H) if w["v"] is not None else None
    n = w["dense"]["n"]
    r["dgu"], r["ddown"] = deq(w["dense"]["gu"], 2 * n, H), deq(w["dense"]["down"], H, n)
    REF[li] = r


_XC = {}


def expert_ref(w, e, xe):
    key = (id(w), e)
    if key not in _XC:
        m = w["experts"][e]
        parts = [(deq(m["gu"], 128, H), deq(m["down"], H, 64))]
        if e in w["extra"]:
            x = w["extra"][e]
            parts.append((deq(x["gu"], 128, H), deq(x["down"], H, 64)))
        _XC[key] = parts
    parts = _XC[key]
    y = torch.zeros(H)
    for gu, dn in parts:
        hgu = gu @ xe
        y += dn @ (gelu(hgu[:64]) * hgu[64:])
    return y


def rd(addr, n):
    return T._read_bf(shard.ue, addr, (n,)).float()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def put_parts(vec):
    shard.ue.dma_write(T.DMA_DEVICE_H2C, shard.PARTS, T._bf(vec), vec.numel() * 2)


torch.manual_seed(0)
caches = {li: ([], []) for li in LAYERS}
worst = {}


def note(name, v):
    worst[name] = max(worst.get(name, 0.0), v)
    return v


HEADW = None
for pos in range(NPOS):
    full = pos % SAMPLE == 0 or pos >= NPOS - 3
    x = (torch.randn(H) * 0.5).to(torch.bfloat16).float()
    shard.set_position(pos, x.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
    X = x.clone()
    line = ["pos %d" % pos]
    for i, li in enumerate(LAYERS):
        w, r = W[li], REF[li]
        sA, sR, sD, sE = 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3
        us_a = shard.run(shard.seg_start[sA])
        if i > 0:                                     # the A segment first finished the previous layer's FFN
            note("X after FFN", rel(rd(shard.X, H), X))
            line.append("X(ffn) %.4f" % rel(rd(shard.X, H), X))
        share_a = rd(shard.SHARE_A, H)
        if r["nq"]:
            hd, nq = r["hd"], r["nq"]
            kind = "s" if w["sliding"] else "g"
            xn = rms(X, w["ln_in"])
            q = (r["q"] @ xn).view(nq, hd)
            kraw = r["k"] @ xn
            vraw = r["v"] @ xn if r["v"] is not None else kraw
            qn, kn, vn = rms(q, w["q_norm"]), rms(kraw, w["k_norm"]), rms(vraw)
            k = rope_hf(kn, pos, kind)
            Kc, Vc = caches[li]
            Kc.append(k); Vc.append(vn)
            lo = max(0, pos - cfg["sliding_window"] + 1) if w["sliding"] else 0
            Kk, Vv = torch.stack(Kc[lo:]), torch.stack(Vc[lo:])
            outs = [torch.softmax(Kk @ rope_hf(qn[h], pos, kind), 0) @ Vv for h in range(nq)]
            ref_a = r["o"] @ torch.cat(outs)
            line.append("A%d %.0fus %.4f" % (li, us_a, note("A", rel(share_a, ref_a))))
        put_parts(share_a)
        us_r = shard.run(shard.seg_start[sR])
        X = X + rms(share_a, w["ln_post_attn"])
        note("X after attn", rel(rd(shard.X, H), X))
        logits = rd(shard.LOGITS, 128)
        ref_l = f(w["router_w"]) @ rms(X)
        idx = torch.topk(logits, 8).indices.tolist()
        ref_idx = torch.topk(ref_l, 8).indices.tolist()
        line.append("R%d %.0fus %.4f top8 %s" % (li, us_r, note("R", rel(logits, ref_l)),
                                                 "same" if sorted(idx) == sorted(ref_idx) else "DIFF"))
        lg_bf = [int(v) for v in torch.tensor([float(logits[e]) for e in idx]).to(torch.bfloat16).view(torch.int16)]
        shard.set_experts_bf(i, idx, lg_bf)
        us_e = shard.run(shard.seg_start[sD]) + shard.run(shard.seg_start[sE])
        share_e = rd(shard.SHARE_E, 2 * H)
        if full:
            xd, xe = rms(X, w["ln_pre_ff"]), rms(X, w["ln_pre_ff2"])
            n = w["dense"]["n"]
            gu = r["dgu"] @ xd
            ref_d = r["ddown"] @ (gelu(gu[:n]) * gu[n:])
            wts = torch.softmax(torch.tensor([float(logits[e]) for e in idx]).to(torch.bfloat16).float(), 0)
            ref_m = sum(wi * expert_ref(w, e, xe) for wi, e in zip(wts, idx))
            line.append("E%d %.0fus dense %.4f moe %.4f" % (li, us_e, note("E dense", rel(share_e[:H], ref_d)),
                                                             note("E moe", rel(share_e[H:], ref_m))))
        put_parts(share_e)
        h = rms(rms(share_e[:H], w["ln_post_ff1"]) + rms(share_e[H:], w["ln_post_ff2"]), w["ln_post_ff"])
        X = (X + h) * w["layer_scalar"]
    us_h = shard.run(shard.seg_start[4 * len(LAYERS)])
    note("X after FFN", rel(rd(shard.X, H), X))
    if full:
        if HEADW is None:
            HEADW = deq(torch.load(os.path.join(cdir, "lm_head.pt"), weights_only=False), vocab[1] - vocab[0], H)
        head = rd(shard.HEAD_OUT, vocab[1] - vocab[0])
        ref_h = HEADW @ rms(X, torch.load(os.path.join(WORK, "weights", "final_norm.pt")))
        line.append("H %.0fus %.4f argmax %s" % (us_h, note("H", rel(head, ref_h)),
                                                 "same" if int(head.argmax()) == int(ref_h.argmax()) else "DIFF"))
        print(" | ".join(line), flush=True)
print("WORST", " ".join("%s=%.4f" % kv for kv in worst.items()), flush=True)
