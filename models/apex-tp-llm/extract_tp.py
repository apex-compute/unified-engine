"""Slice a Qwen3.5-family model for tensor-parallel inference: every card gets a slice of EVERY layer.

Per layer a card holds
  linear attention : some key heads (with the value heads they serve) -> rows of in_proj_qkv / in_proj_z /
                     in_proj_a / in_proj_b / conv1d, columns of out_proj
  full attention   : some query heads plus the one KV head they attend with -> rows of q/k/v_proj,
                     columns of o_proj
  MLP              : a block of the intermediate neurons -> rows of gate/up, columns of down
and a block of vocabulary rows of the LM head. A slice is itself a (narrower) layer, so the port's
own extractors quantize it unchanged. Host-side preparation only.

    Q_MODEL=q35_9b python extract_tp.py 11      ->  $APEX_TP_WORK/models/q35_9b/tp11/card_00 .. card_10, plan.json
    Q_MODEL=q35_9b python extract_tp.py 11 v2   ->  the same slices under tp11v2
"""
import glob
import importlib.util
import json
import os
import sys
import time
import types

import torch
from safetensors import safe_open

P = int(sys.argv[1])
BASE = os.path.join(os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp")), "models", os.environ.get("Q_MODEL", "q35_9b"))
MODEL = os.path.dirname(sorted(glob.glob(os.path.join(BASE, "*_bin", "*", "config.json")))[0])
WDIR = os.path.join(BASE, "weights")          # card-independent pieces: final norm, embedding table
# Several copies may run side by side on disjoint layer ranges (TP_LAYERS=lo:hi); one of them
# (TP_HEAD=1, the default when no range is given) also writes the LM head and the shared pieces.
_lo, _hi = (int(v) for v in os.environ.get("TP_LAYERS", "0:100000").split(":"))
DO_HEAD = os.environ.get("TP_HEAD", "0" if "TP_LAYERS" in os.environ else "1") == "1"
OUT = os.path.join(BASE, "tp%d%s" % (P, sys.argv[2] if len(sys.argv) > 2 else ""))

spec = importlib.util.spec_from_file_location("q35_9b_port", os.path.join(BASE, "qwen3.5_2b_test.py"))
T = importlib.util.module_from_spec(spec)
sys.modules["q35_9b_port"] = T
spec.loader.exec_module(T)

cfg = json.load(open(os.path.join(BASE, "qwen3.5_2b_config.json")))
fi = cfg["file_info"]
H, MLP, VOCAB = fi["hidden_size"], fi["mlp_elements"], fi["embedding_vocab"]
NQ, NKV, HD = fi["full_attn_num_heads"], fi["full_attn_num_kv_heads"], fi["full_attn_head_dim"]
NVH, NKH = fi["linear_attn_num_v_heads"], fi["linear_attn_num_k_heads"]
DK, DV = fi["linear_attn_head_k_dim"], fi["linear_attn_head_v_dim"]
KEY_DIM, VAL_DIM = fi["linear_attn_key_dim"], fi["linear_attn_value_dim"]
REP = NVH // NKH
n_layers = fi["num_layers"]
full_layers = set(cfg["model"]["full_attn_layer_indices"])


def make_plan():
    # Linear attention is split by VALUE head (finer than key heads, so the cards come out more
    # even). A card computes the query/key rows of whichever key head each of its value heads
    # uses; a key head whose value heads land on two cards is simply computed on both.
    base, extra = divmod(NVH, P)
    assert base >= 1, "more cards than value heads"
    vh_n = [base + (1 if i < extra else 0) for i in range(P)]
    vh0 = [sum(vh_n[:i]) for i in range(P)]
    # Full attention: whole query heads, packed so each card's query heads share one KV head.
    base, extra = divmod(NQ, P)
    kh_n = [base + (1 if i < extra else 0) for i in range(P)]
    cap = NQ // NKV
    groups = [[] for _ in range(NKV)]
    try:
        for i in sorted(range(P), key=lambda i: -kh_n[i]):
            g = next(g for g in range(NKV) if sum(kh_n[j] for j in groups[g]) + kh_n[i] <= cap)
            groups[g].append(i)
        assert all(sum(kh_n[j] for j in m) == cap for m in groups)
    except (StopIteration, AssertionError):
        # Equal query-head counts do not pack (e.g. 16 heads, 4 KV groups, 5 cards): spread the
        # cards over the KV groups first, then split each group's query heads among its cards.
        assert P >= NKV, "fewer cards than KV heads"
        gb, ge = divmod(P, NKV)
        per_group = [gb + (1 if g < ge else 0) for g in range(NKV)]
        assert max(per_group) <= cap, "more cards in a KV group than query heads"
        groups, kh_n, i = [], [0] * P, 0
        for g in range(NKV):
            hb, he = divmod(cap, per_group[g])
            members = list(range(i, i + per_group[g]))
            for r, j in enumerate(members):
                kh_n[j] = hb + (1 if r < he else 0)
            groups.append(members)
            i += per_group[g]
    qh = {}
    for g, members in enumerate(groups):
        assert sum(kh_n[j] for j in members) == cap, "query heads do not pack into KV groups"
        h = g * cap
        for j in members:
            qh[j] = (g, list(range(h, h + kh_n[j])))
            h += kh_n[j]
    # MLP neurons (64-blocks): equal shares. The cards meet after every half layer, so each half
    # is as slow as its slowest card; balancing the MLP against the attention would not help.
    mb, me = divmod(MLP // 64, P)
    blocks = [mb + (1 if i >= P - me else 0) for i in range(P)]     # the extra blocks go to the cards with fewer heads
    vb, ve = divmod(VOCAB // 64, P)
    vblocks = [vb + (1 if i < ve else 0) for i in range(P)]
    cards = []
    for i in range(P):
        m0, v0 = 64 * sum(blocks[:i]), 64 * sum(vblocks[:i])
        cards.append({"vh": [vh0[i], vh0[i] + vh_n[i]], "kv": qh[i][0], "qh": qh[i][1],
                      "mlp": [m0, m0 + 64 * blocks[i]], "vocab": [v0, v0 + 64 * vblocks[i]]})
    return cards


cards = make_plan()
os.makedirs(OUT, exist_ok=True)
for k, c in enumerate(cards):
    d = os.path.join(OUT, "card_%02d" % k)
    os.makedirs(d, exist_ok=True)
    n_vh, n_q = c["vh"][1] - c["vh"][0], len(c["qh"])
    lc = json.loads(json.dumps(cfg))
    lc["file_info"].update({
        "mlp_elements": c["mlp"][1] - c["mlp"][0], "embedding_vocab": c["vocab"][1] - c["vocab"][0],
        "full_attn_num_heads": n_q, "full_attn_num_kv_heads": 1,
        "full_attn_q_size": n_q * HD, "full_attn_kv_size": HD,
        # one (duplicated) key head per value head on the card
        "linear_attn_num_k_heads": n_vh, "linear_attn_num_v_heads": n_vh, "linear_attn_num_ab_heads": n_vh,
        "linear_attn_key_dim": n_vh * DK, "linear_attn_value_dim": n_vh * DV,
        "linear_attn_conv_dim": 2 * n_vh * DK + n_vh * DV})
    json.dump(lc, open(os.path.join(d, "config.json"), "w"), indent=1)
    print("PLAN|card %d|value heads %s|query heads %s (KV %d)|MLP %s|vocabulary %s" % (
        k, c["vh"], c["qh"], c["kv"], c["mlp"], c["vocab"]), flush=True)
json.dump({"P": P, "cards": cards, "hidden": H, "layers": n_layers}, open(os.path.join(OUT, "plan.json"), "w"), indent=1)

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_open = {}


def get(key):
    f = index[key]
    if f not in _open:
        _open[f] = safe_open(os.path.join(MODEL, f), "pt")
    return _open[f].get_tensor(key)


def NS(w):
    return types.SimpleNamespace(weight=w)


def rng(a, b, unit, off=0):
    return torch.arange(off + a * unit, off + b * unit)


t_all = time.time()
for i in range(max(0, _lo), min(n_layers, _hi)):
    if all(os.path.exists(os.path.join(OUT, "card_%02d" % k, "layer_%02d.pt" % i)) for k in range(P)):
        continue
    t0 = time.time()
    p = "model.language_model.layers.%d." % i
    ln_in, ln_post = get(p + "input_layernorm.weight"), get(p + "post_attention_layernorm.weight")
    gate, up, down = get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight"), get(p + "mlp.down_proj.weight")
    if i in full_layers:
        a = p + "self_attn."
        A = {n: get(a + n + ".weight") for n in ("q_norm", "k_norm", "q_proj", "k_proj", "v_proj", "o_proj")}
    else:
        a = p + "linear_attn."
        A = {n: get(a + n + ".weight") for n in ("norm", "in_proj_qkv", "in_proj_z", "out_proj", "in_proj_b",
                                                 "in_proj_a", "conv1d")}
        A["A_log"], A["dt_bias"] = get(a + "A_log"), get(a + "dt_bias")
    for k, c in enumerate(cards):
        m0, m1 = c["mlp"]
        lyr = types.SimpleNamespace(
            input_layernorm=NS(ln_in), post_attention_layernorm=NS(ln_post),
            mlp=types.SimpleNamespace(gate_proj=NS(gate[m0:m1]), up_proj=NS(up[m0:m1]),
                                      down_proj=NS(down[:, m0:m1].contiguous())))
        if i in full_layers:
            q_rows = torch.cat([rng(h, h + 1, 2 * HD) for h in c["qh"]])
            o_cols = torch.cat([rng(h, h + 1, HD) for h in c["qh"]])
            kv_rows = rng(c["kv"], c["kv"] + 1, HD)
            lyr.self_attn = types.SimpleNamespace(
                q_norm=NS(A["q_norm"]), k_norm=NS(A["k_norm"]), q_proj=NS(A["q_proj"][q_rows]),
                k_proj=NS(A["k_proj"][kv_rows]), v_proj=NS(A["v_proj"][kv_rows]),
                o_proj=NS(A["o_proj"][:, o_cols].contiguous()))
            ex = T._extract_full_attn_weights(lyr)
        else:
            v0, v1 = c["vh"]
            v_heads = torch.arange(v0, v1)
            v_rows = rng(v0, v1, DV)
            key_rows = torch.cat([rng(v // REP, v // REP + 1, DK) for v in range(v0, v1)])
            qkv_rows = torch.cat([key_rows, key_rows + KEY_DIM, v_rows + 2 * KEY_DIM])
            la = types.SimpleNamespace(
                norm=NS(A["norm"]), in_proj_qkv=NS(A["in_proj_qkv"][qkv_rows]), in_proj_z=NS(A["in_proj_z"][v_rows]),
                out_proj=NS(A["out_proj"][:, v_rows].contiguous()), in_proj_b=NS(A["in_proj_b"][v_heads]),
                in_proj_a=NS(A["in_proj_a"][v_heads]), conv1d=NS(A["conv1d"][qkv_rows]))
            la.A_log, la.dt_bias = A["A_log"][v_heads], A["dt_bias"][v_heads]
            lyr.linear_attn = la
            ex = T._extract_linear_attn_weights(lyr)
        dst = os.path.join(OUT, "card_%02d" % k, "layer_%02d.pt" % i)
        torch.save(ex, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    print("LAYER|%d|%s|%.1f s" % (i, "full" if i in full_layers else "linear", time.time() - t0), flush=True)

if DO_HEAD and not all(os.path.exists(os.path.join(OUT, "card_%02d" % k, "lm_head.pt")) for k in range(P)):
    t0 = time.time()
    head = get("lm_head.weight")
    for k, c in enumerate(cards):
        v0, v1 = c["vocab"]
        dst = os.path.join(OUT, "card_%02d" % k, "lm_head.pt")
        torch.save(T.quantize_lm_head_fp4(head[v0:v1].to(torch.bfloat16)), dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    print("LM_HEAD|%.1f s" % (time.time() - t0), flush=True)
if DO_HEAD:
    os.makedirs(WDIR, exist_ok=True)
    dst = os.path.join(WDIR, "final_norm.pt")
    if not os.path.exists(dst):
        torch.save((get("model.language_model.norm.weight").float() + 1.0).to(torch.bfloat16), dst)
    dst = os.path.join(WDIR, "embed.bin")
    if not os.path.exists(dst):
        e = get("model.language_model.embed_tokens.weight").to(torch.bfloat16).contiguous()
        with open(dst + ".tmp", "wb") as f:
            f.write(e.view(torch.int16).numpy().tobytes())
        os.replace(dst + ".tmp", dst)
        json.dump({"shape": list(e.shape)}, open(os.path.join(WDIR, "embed.json"), "w"))
        print("EMBED|%s" % list(e.shape), flush=True)
print("DONE|%.0f s" % (time.time() - t_all), flush=True)
