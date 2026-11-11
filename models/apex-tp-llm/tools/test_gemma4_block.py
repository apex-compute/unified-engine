"""Unit test of the image-block program on one engine: one shard, layers 0 (sliding) and 5 (global).
PRE text tokens (normal program), a block of NB image tokens (block program), POST text tokens (normal program).
Every segment is compared with a torch reference from the SAME sliced weights that applies transformers' masks:
sliding  kv > q - window AND (kv <= q OR both in the image block),   global  kv <= q.
WIN shrinks the sliding window so that a short test wraps the ring and lets the block cross the window edge.
The reference follows the card's own shares (like test_gemma4.py), so each number is that segment's error alone.
Test only: the reference never runs during inference."""
import json
import os
import sys

import torch
torch.set_num_threads(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4"))
sys.path.insert(1, os.path.expanduser("~/unified-engine"))
import gemma4_card as G  # noqa: E402
import quant_lib as Q  # noqa: E402

WORK = os.path.expanduser("~/apex_tp/models/gemma4_26b")
TP = os.path.join(WORK, "tp10")
plan = json.load(open(os.path.join(TP, "plan.json")))
cfg = plan["config"]
WIN = int(os.environ.get("WIN", "128"))
cfg["sliding_window"] = WIN
H, EPS = cfg["hidden_size"], cfg["rms_norm_eps"]
SHARD = int(os.environ.get("SHARD", "0"))
LAYERS = [int(v) for v in os.environ.get("LAYERS", "0,5").split(",")]
PRE, NB, POST = int(os.environ.get("PRE", "250")), int(os.environ.get("NB", "90")), int(os.environ.get("POST", "4"))
MAXB = int(os.environ.get("MAXB", "96"))
SAMPLE = int(os.environ.get("SAMPLE", "8"))       # FFN reference every SAMPLE tokens (+ the last ones)
cdir = os.path.join(TP, "card_%02d" % SHARD)

T, qcfg = G.load_base_port("gemma4_block_test_port")
shard = G.Gemma4Shard(T, qcfg, sys.argv[1] if len(sys.argv) > 1 else "xdma0", 0, cdir, os.path.join(WORK, "weights"),
                      plan, max_ctx=int(os.environ.get("CTX", "512")), layers=LAYERS, max_block=MAXB)
with T._quiet():
    shard.prepare()
vocab = plan["cards"][SHARD]["vocab"]
n_seg = shard.compile(parts_n=1, vocab_n=vocab[1] - vocab[0])
with T._quiet():
    n_bseg = shard.compile_block()
print("compiled: %d segments, block program %d segments %s, ring %d rows, window %d" % (
    n_seg, n_bseg, "".join(shard.bseg_kind), shard.RING, WIN), flush=True)


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
    hd = x.shape[-1]
    rp = cfg["rope_parameters"]["sliding_attention" if kind == "s" else "full_attention"]
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
    y = torch.zeros(H)
    for gu, dn in _XC[key]:
        hgu = gu @ xe
        y += dn @ (gelu(hgu[:64]) * hgu[64:])
    return y


def rd(addr, n):
    return T._read_bf(shard.ue, addr, (n,)).float()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def put_parts(vec):
    shard.ue.dma_write(T.DMA_DEVICE_H2C, shard.PARTS, T._bf(vec), vec.numel() * 2)


BLOCK = (PRE, PRE + NB - 1)
caches = {li: {} for li in LAYERS}                 # layer -> position -> (k after RoPE, v)
worst = {}


def note(name, v):
    worst[name] = max(worst.get(name, 0.0), v)
    return v


def kv_ref(li, X, pos):
    w, r = W[li], REF[li]
    kind = "s" if w["sliding"] else "g"
    xn = rms(X, w["ln_in"])
    kraw = r["k"] @ xn
    vraw = r["v"] @ xn if r["v"] is not None else kraw
    caches[li][pos] = (rope_hf(rms(kraw, w["k_norm"]), pos, kind), rms(vraw))


def attn_ref(li, X, pos):
    """The share of this shard's heads; K / V of every visible position must be in caches[li]."""
    w, r = W[li], REF[li]
    hd, nq = r["hd"], r["nq"]
    kind = "s" if w["sliding"] else "g"
    q = (r["q"] @ rms(X, w["ln_in"])).view(nq, hd)
    qn = rms(q, w["q_norm"])
    in_blk = BLOCK[0] <= pos <= BLOCK[1]
    vis = []
    for kv in sorted(caches[li]):
        if w["sliding"]:
            ok = kv > pos - WIN and (kv <= pos or (in_blk and BLOCK[0] <= kv <= BLOCK[1]))
        else:
            ok = kv <= pos
        if ok:
            vis.append(kv)
    Kk = torch.stack([caches[li][p][0] for p in vis])
    Vv = torch.stack([caches[li][p][1] for p in vis])
    outs = [torch.softmax(Kk @ rope_hf(qn[h], pos, kind), 0) @ Vv for h in range(nq)]
    return r["o"] @ torch.cat(outs), len(vis)


def ffn_check(li, X, logits, idx, share_e, tag):
    w, r = W[li], REF[li]
    xd, xe = rms(X, w["ln_pre_ff"]), rms(X, w["ln_pre_ff2"])
    n = w["dense"]["n"]
    gu = r["dgu"] @ xd
    ref_d = r["ddown"] @ (gelu(gu[:n]) * gu[n:])
    wts = torch.softmax(torch.tensor([float(logits[e]) for e in idx]).to(torch.bfloat16).float(), 0)
    ref_m = sum(wi * expert_ref(w, e, xe) for wi, e in zip(wts, idx))
    return "E dense %.4f moe %.4f" % (note(tag + " E dense", rel(share_e[:H], ref_d)),
                                      note(tag + " E moe", rel(share_e[H:], ref_m)))


def router_pick(i, li, X, block):
    logits = rd(shard.LOGITS, 128)
    w = W[li]
    ref_l = f(w["router_w"]) @ rms(X)
    idx = torch.topk(logits, 8).indices.tolist()
    same = sorted(idx) == sorted(torch.topk(ref_l, 8).indices.tolist())
    lg_bf = [int(v) for v in torch.tensor([float(logits[e]) for e in idx]).to(torch.bfloat16).view(torch.int16)]
    shard.set_experts_bf(i, idx, lg_bf, block=block)
    return logits, idx, rel(logits, ref_l), same


def ffn_update(li, X, share_e):
    w = W[li]
    h = rms(rms(share_e[:H], w["ln_post_ff1"]) + rms(share_e[H:], w["ln_post_ff2"]), w["ln_post_ff"])
    return (X + h) * w["layer_scalar"]


torch.manual_seed(0)
diffs = [0]


def text_token(pos, tag):
    """One token through the normal program (as test_gemma4.py)."""
    x = (torch.randn(H) * 0.5).to(torch.bfloat16).float()
    shard.set_position(pos, x.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
    X = x.clone()
    line = ["%s pos %d" % (tag, pos)]
    full = pos % SAMPLE == 0 or pos >= PRE + NB + POST - 2
    for i, li in enumerate(LAYERS):
        w, r = W[li], REF[li]
        shard.run(shard.seg_start[4 * i])
        if i > 0:
            line.append("X %.4f" % note(tag + " X", rel(rd(shard.X, H), X)))
        share_a = rd(shard.SHARE_A, H)
        if r["nq"]:
            kv_ref(li, X, pos)
            ref_a, nvis = attn_ref(li, X, pos)
            line.append("A%d(%d) %.4f" % (li, nvis, note(tag + " A", rel(share_a, ref_a))))
        put_parts(share_a)
        shard.run(shard.seg_start[4 * i + 1])
        X = X + rms(share_a, w["ln_post_attn"])
        logits, idx, er, same = router_pick(i, li, X, False)
        diffs[0] += not same
        shard.run(shard.seg_start[4 * i + 2])
        shard.run(shard.seg_start[4 * i + 3])
        share_e = rd(shard.SHARE_E, 2 * H)
        if full:
            line.append(ffn_check(li, X, logits, idx, share_e, tag))
        put_parts(share_e)
        X = ffn_update(li, X, share_e)
    shard.run(shard.seg_start[4 * len(LAYERS)])
    line.append("X %.4f" % note(tag + " X", rel(rd(shard.X, H), X)))
    if full:
        print(" | ".join(line), flush=True)


for pos in range(PRE):
    text_token(pos, "pre")

# ---------------- the block, layer by layer
p0 = PRE
xs = [(torch.randn(H) * 0.5).to(torch.bfloat16).float() for _ in range(NB)]
shard.block_begin(p0, NB, b"".join(x.to(torch.bfloat16).view(torch.int16).numpy().tobytes() for x in xs))
Xs = [x.clone() for x in xs]
seg = {(k, i): s for s, (k, i) in enumerate(zip(shard.bseg_kind, shard.bseg_layer))}
for t in range(NB):
    shard.block_slots(seg[("K", -1)], t)
    shard.run(shard.bseg_start[seg[("K", -1)]])
for i, li in enumerate(LAYERS):
    w, r = W[li], REF[li]
    if r["nq"]:
        for t in range(NB):                                  # the card wrote these in K / the previous F
            kv_ref(li, Xs[t], p0 + t)
    for t in range(NB):
        pos = p0 + t
        full = t % SAMPLE == 0 or t >= NB - 2
        line = ["blk t %d pos %d" % (t, pos)]
        sB = seg[("B", i)]
        shard.block_slots(sB, t)
        shard.block_mask(sB, t)
        shard.run(shard.bseg_start[sB])
        line.append("XB %.4f" % note("blk X in", rel(rd(shard.XB, H), Xs[t])))
        share_a = rd(shard.SHARE_A, H)
        if r["nq"]:
            ref_a, nvis = attn_ref(li, Xs[t], pos)
            line.append("A%d(%d) %.4f" % (li, nvis, note("blk A", rel(share_a, ref_a))))
        put_parts(share_a)
        shard.run(shard.bseg_start[seg[("R", i)]])
        Xs[t] = Xs[t] + rms(share_a, w["ln_post_attn"])
        logits, idx, er, same = router_pick(i, li, Xs[t], True)
        diffs[0] += not same
        shard.run(shard.bseg_start[seg[("D", i)]])
        shard.run(shard.bseg_start[seg[("E", i)]])
        share_e = rd(shard.SHARE_E, 2 * H)
        if full:
            line.append("R %.4f %s" % (note("blk R", er), "same" if same else "DIFF"))
            line.append(ffn_check(li, Xs[t], logits, idx, share_e, "blk"))
        put_parts(share_e)
        shard.run(shard.bseg_start[seg[("F", i)]])
        Xs[t] = ffn_update(li, Xs[t], share_e)
        line.append("XS %.4f" % note("blk X out", rel(rd(shard.XS + t * H * 2, H), Xs[t])))
        if full:
            print(" | ".join(line), flush=True)
shard.block_end()

for pos in range(PRE + NB, PRE + NB + POST):
    text_token(pos, "post")
print("PICK|router top-8 differed from the reference in %d layer-tokens" % diffs[0])
print("WORST", " ".join("%s=%.4f" % kv for kv in worst.items()), flush=True)
