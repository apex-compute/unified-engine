"""Slice google/gemma-4-26b-a4b-it (text part) for P = 10 tensor-parallel shards (host-side preparation only).

Shard k is engine k % 2 of card k // 2 (cards in PCI order; card 4 is the x1 card).

Per layer:
  attention : shards 0-7 only. Sliding layers: KV head k and its 2 query heads (2k, 2k+1).
              Global layers (2 KV heads, K = V): query heads 2k, 2k+1 and KV head k // 4.
              Shards 8-9 (the x1 card) have no attention work.
  router    : a full bf16 copy, with router.scale * hidden**-0.5 folded into its columns.
  dense FFN : 33 blocks of 64 neurons; shards 0-2 hold 4 blocks, the others 3.
  experts   : 11 blocks of 64 neurons per expert. Shard k holds block k of every expert. Block 10 of
              expert e goes to one of the shards with 3 dense blocks (3-9), rotating with e and the layer.
              router.per_expert_scale is folded into each expert's down matrix.
  norms     : kept as bf16 vectors (applied on the card); layer_scalar kept as a number.
LM head (tied to the embedding): 64-row vocabulary blocks over all shards.
Host RAM: the embedding table, pre-multiplied by sqrt(hidden) in bf16 (as the model does), and the final norm.

Weights become the card's FP4 (E2M1, one bf16 scale per 64 values along K).

    python extract_gemma4.py               ->  $GEMMA4_WORK/tp10/card_00..09 + plan.json + weights/
    GEMMA4_LAYERS=0:8 python extract_gemma4.py   (several copies on disjoint layer ranges; head with GEMMA4_HEAD=1)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/unified-engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_lib as Q  # noqa: E402
from st_read import SafeDir  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
P = 10
MODEL = os.path.expanduser(os.environ.get("GEMMA4_MODEL", "~/apex_exp/models/gemma4_26b/hf/gemma-4-26b-a4b-it"))
WORK = os.path.expanduser(os.environ.get("GEMMA4_WORK", os.path.join(os.environ.get("APEX_TP_WORK", "~/apex_tp"), "models", "gemma4_26b")))
OUT = os.path.join(WORK, "tp%d" % P)
WDIR = os.path.join(WORK, "weights")
cfg_all = json.load(open(os.path.join(MODEL, "config.json")))
cfg = cfg_all["text_config"]
H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
NQ, NKV, NKVG = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_global_key_value_heads"]
HD, GHD = cfg["head_dim"], cfg["global_head_dim"]
NE, IE, ID = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["intermediate_size"]
VOCAB = cfg["vocab_size"]
BLK = 64
NEB, NDB = IE // BLK, ID // BLK                     # 11 expert blocks, 33 dense blocks
ATT_SHARDS = 8
_lo, _hi = (int(v) for v in os.environ.get("GEMMA4_LAYERS", "0:100000").split(":"))
DO_HEAD = os.environ.get("GEMMA4_HEAD", "0" if "GEMMA4_LAYERS" in os.environ else "1") == "1"
assert IE % BLK == 0 and ID % BLK == 0 and H % BLK == 0 and VOCAB % BLK == 0
assert NQ == 2 * ATT_SHARDS and NKV == ATT_SHARDS and NEB == P + 1

S = SafeDir(MODEL)
PRE = "model.language_model."


def get(key):
    return S.get(PRE + key)


def fp4(w):
    """[N, K] -> (scales bf16 [N*K/64], packed uint8 [N*K/2]); K must be a multiple of 64."""
    assert w.dim() == 2 and w.shape[1] % 64 == 0, w.shape
    d, s = Q.quantize_if4(w.to(torch.bfloat16).contiguous(), 64, int_variant=False)
    return (torch.frombuffer(bytearray(s), dtype=torch.bfloat16).clone(),
            torch.frombuffer(bytearray(d), dtype=torch.uint8).clone())


def split_blocks(total, parts):
    b, e = divmod(total, parts)
    sizes = [b + (1 if i < e else 0) for i in range(parts)]
    starts = [sum(sizes[:i]) for i in range(parts)]
    return [(s, s + n) for s, n in zip(starts, sizes)]


DENSE = split_blocks(NDB, P)                        # shards 0-2: 4 blocks, 3-9: 3 blocks
EXTRA_OWNERS = [k for k in range(P) if DENSE[k][1] - DENSE[k][0] == min(b - a for a, b in DENSE)]


def extra_owner(e, li):
    """Shard that holds block 10 of expert e in layer li."""
    return EXTRA_OWNERS[(e + li) % len(EXTRA_OWNERS)]


def make_plan():
    vb = split_blocks(VOCAB // BLK, P)
    cards = []
    for k in range(P):
        att = k < ATT_SHARDS
        cards.append({
            "q_heads": [2 * k, 2 * k + 1] if att else [],
            "kv_head_sliding": k if att else None,
            "kv_head_global": (2 * k) // (NQ // NKVG) if att else None,
            "dense_blocks": list(DENSE[k]),
            "expert_block": k,
            "vocab": [BLK * vb[k][0], BLK * vb[k][1]],
        })
    return cards


cards = make_plan()
os.makedirs(OUT, exist_ok=True)
for k, c in enumerate(cards):
    os.makedirs(os.path.join(OUT, "card_%02d" % k), exist_ok=True)
    print("PLAN|shard %d|q heads %s|kv sliding %s global %s|dense blocks %s|expert block %d|vocab %s" % (
        k, c["q_heads"], c["kv_head_sliding"], c["kv_head_global"], c["dense_blocks"], c["expert_block"],
        c["vocab"]), flush=True)
json.dump({"P": P, "cards": cards, "hidden": H, "layers": L, "block": BLK, "extra_owners": EXTRA_OWNERS,
           "config": cfg}, open(os.path.join(OUT, "plan.json"), "w"), indent=1)


def rows(heads, hd):
    return torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in heads])


t_all = time.time()
for li in range(max(0, _lo), min(L, _hi)):
    if all(os.path.exists(os.path.join(OUT, "card_%02d" % k, "layer_%02d.pt" % li)) for k in range(P)):
        continue
    t0 = time.time()
    p = "layers.%d." % li
    sliding = cfg["layer_types"][li] == "sliding_attention"
    hd = HD if sliding else GHD
    wq, wk, wo = get(p + "self_attn.q_proj.weight"), get(p + "self_attn.k_proj.weight"), get(p + "self_attn.o_proj.weight")
    wv = get(p + "self_attn.v_proj.weight") if sliding else None          # global layers: V comes from k_proj (K = V)
    gu = get(p + "experts.gate_up_proj")            # [E, 2*IE, H]: gate rows 0..IE-1, up rows IE..2*IE-1
    dn = get(p + "experts.down_proj")               # [E, H, IE]
    pes = get(p + "router.per_expert_scale").float()
    dg, du, dd = get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight"), get(p + "mlp.down_proj.weight")
    router = (get(p + "router.proj.weight").float()
              * (get(p + "router.scale").float() * H ** -0.5)[None, :]).to(torch.bfloat16)
    bf = lambda n: get(p + n).to(torch.bfloat16)   # noqa: E731
    common = {"sliding": sliding, "head_dim": hd,
              "layer_scalar": float(get(p + "layer_scalar").float()[0]),
              "ln_in": bf("input_layernorm.weight"), "ln_post_attn": bf("post_attention_layernorm.weight"),
              "ln_pre_ff": bf("pre_feedforward_layernorm.weight"), "ln_pre_ff2": bf("pre_feedforward_layernorm_2.weight"),
              "ln_post_ff1": bf("post_feedforward_layernorm_1.weight"),
              "ln_post_ff2": bf("post_feedforward_layernorm_2.weight"),
              "ln_post_ff": bf("post_feedforward_layernorm.weight"),
              "q_norm": bf("self_attn.q_norm.weight"), "k_norm": bf("self_attn.k_norm.weight"),
              "router_w": router}

    def expert_block(e, b):
        n0, n1 = b * BLK, (b + 1) * BLK
        g, u = gu[e][n0:n1], gu[e][IE + n0:IE + n1]                      # [64, H] each
        return {"gu": fp4(torch.cat([g, u], 0)),                          # [128, H]: gate rows, then up rows
                "down": fp4((dn[e][:, n0:n1].float() * pes[e]).to(torch.bfloat16))}   # [H, 64]

    for k, c in enumerate(cards):
        layer = dict(common)
        if c["q_heads"]:
            qr = rows(c["q_heads"], hd)
            kvh = c["kv_head_sliding"] if sliding else c["kv_head_global"]
            kr = rows([kvh], hd)
            layer.update({"q": fp4(wq[qr]), "k": fp4(wk[kr]), "v": fp4(wv[kr]) if sliding else None,
                          "o": fp4(wo[:, qr]), "q_heads": c["q_heads"], "kv_head": kvh})
        else:
            layer.update({"q": None, "k": None, "v": None, "o": None, "q_heads": [], "kv_head": None})
        b0, b1 = c["dense_blocks"]
        n0, n1 = b0 * BLK, b1 * BLK
        layer["dense"] = {"gu": fp4(torch.cat([dg[n0:n1], du[n0:n1]], 0)),    # [2n, H]: gate rows, then up rows
                          "down": fp4(dd[:, n0:n1]), "n": n1 - n0}
        layer["experts"] = [expert_block(e, c["expert_block"]) for e in range(NE)]
        layer["extra"] = {e: expert_block(e, NEB - 1) for e in range(NE) if extra_owner(e, li) == k}
        dst = os.path.join(OUT, "card_%02d" % k, "layer_%02d.pt" % li)
        torch.save(layer, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    del gu, dn
    print("LAYER|%d|%s|%.1f s" % (li, cfg["layer_types"][li], time.time() - t0), flush=True)

if DO_HEAD:
    t0 = time.time()
    emb = get("embed_tokens.weight")                 # [VOCAB, H] bf16, tied LM head
    for k, c in enumerate(cards):
        v0, v1 = c["vocab"]
        dst = os.path.join(OUT, "card_%02d" % k, "lm_head.pt")
        if not os.path.exists(dst):
            torch.save(fp4(emb[v0:v1]), dst + ".tmp")
            os.replace(dst + ".tmp", dst)
    os.makedirs(WDIR, exist_ok=True)
    torch.save(get("norm.weight").to(torch.bfloat16), os.path.join(WDIR, "final_norm.pt"))
    scaled = emb * torch.tensor(H ** 0.5, dtype=torch.bfloat16)       # Gemma4TextScaledWordEmbedding, in bf16
    with open(os.path.join(WDIR, "embed.bin.tmp"), "wb") as f:
        f.write(scaled.contiguous().view(torch.int16).numpy().tobytes())
    os.replace(os.path.join(WDIR, "embed.bin.tmp"), os.path.join(WDIR, "embed.bin"))
    json.dump({"shape": list(scaled.shape), "scale": H ** 0.5, "softcap": cfg["final_logit_softcapping"]},
              open(os.path.join(WDIR, "embed.json"), "w"))
    print("HEAD|%.1f s" % (time.time() - t0), flush=True)
print("DONE|%.0f s" % (time.time() - t_all), flush=True)
