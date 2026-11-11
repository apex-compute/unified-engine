"""Slice openai/gpt-oss-20b for P tensor-parallel shards (host-side preparation only).

Per layer every shard holds:
  attention : its query heads (whole KV groups or an even part of one), their KV head, q/k/v/o
              weights and biases (the o bias only on shard 0), the heads' sinks
  router    : a full copy (32 x 2880, tiny); every shard computes the same 32 logits
  experts   : for all 32 experts, the shard's block of the 2880 intermediate neurons:
              gate and up rows stacked [gate; up] (the checkpoint interleaves them), their biases,
              the matching down columns, and the down bias (shard 0 only; the cards sum the shares)
plus a block of LM-head vocabulary rows. MXFP4 experts are dequantized with transformers' own
converter and requantized to the card's FP4 (E2M1, one bf16 scale per 64 along K).

    python extract_gptoss.py 10            ->  $GPTOSS_WORK/tp10/card_00..09 + plan.json + weights/
    GPTOSS_LAYERS=0:6 python extract_gptoss.py 10   (several copies on disjoint layer ranges)
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import quant_lib as Q  # noqa: E402
from transformers.integrations.mxfp4 import convert_moe_packed_tensors  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
P = int(sys.argv[1])
MODEL = os.path.expanduser(os.environ.get("GPTOSS_MODEL", "~/apex_exp/models/gpt_oss_20b/hf/gpt-oss-20b"))
WORK = os.path.expanduser(os.environ.get("GPTOSS_WORK", "~/apex_tp/models/gptoss20b"))
OUT = os.path.join(WORK, "tp%d" % P)
WDIR = os.path.join(WORK, "weights")
cfg = json.load(open(os.path.join(MODEL, "config.json")))
H, I, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
NQ, NKV, HD, NE = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["num_local_experts"]
VOCAB = cfg["vocab_size"]
_lo, _hi = (int(v) for v in os.environ.get("GPTOSS_LAYERS", "0:100000").split(":"))
DO_HEAD = os.environ.get("GPTOSS_HEAD", "0" if "GPTOSS_LAYERS" in os.environ else "1") == "1"

index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_open = {}


def get(key):
    f = index[key]
    if f not in _open:
        _open[f] = safe_open(os.path.join(MODEL, f), "pt")
    return _open[f].get_tensor(key)


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


def make_plan():
    # experts: 64-neuron blocks of the intermediate size
    nb = split_blocks(I // 64, P)
    # attention: shards spread over the KV groups, each group's query heads split among its shards
    group = NQ // NKV
    if P <= NKV:
        assert NKV % P == 0
        per = NKV // P
        att = [(list(range(k * per, (k + 1) * per)), list(range(k * per * group, (k + 1) * per * group))) for k in range(P)]
    else:
        gb, ge = divmod(P, NKV)
        per_group = [gb + (1 if g < ge else 0) for g in range(NKV)]
        att, k = [], 0
        for g in range(NKV):
            for s0, s1 in split_blocks(group, per_group[g]):
                att.append(([g], list(range(g * group + s0, g * group + s1))))
                k += 1
    vb = split_blocks(VOCAB // 64, P)
    return [{"neurons": [64 * nb[k][0], 64 * nb[k][1]], "kv_heads": att[k][0], "q_heads": att[k][1],
             "vocab": [64 * vb[k][0], 64 * vb[k][1]]} for k in range(P)]


cards = make_plan()
os.makedirs(OUT, exist_ok=True)
for k, c in enumerate(cards):
    os.makedirs(os.path.join(OUT, "card_%02d" % k), exist_ok=True)
    print("PLAN|card %d|q heads %s..%s (%d)|kv heads %s|neurons %s|vocab %s" % (
        k, c["q_heads"][0], c["q_heads"][-1], len(c["q_heads"]), c["kv_heads"], c["neurons"], c["vocab"]), flush=True)
json.dump({"P": P, "cards": cards, "hidden": H, "layers": L, "config": cfg}, open(os.path.join(OUT, "plan.json"), "w"), indent=1)

t_all = time.time()
for li in range(max(0, _lo), min(L, _hi)):
    if all(os.path.exists(os.path.join(OUT, "card_%02d" % k, "layer_%02d.pt" % li)) for k in range(P)):
        continue
    t0 = time.time()
    p = "model.layers.%d." % li
    A = {n: get(p + "self_attn." + n) for n in ("q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias",
                                                  "v_proj.weight", "v_proj.bias", "o_proj.weight", "o_proj.bias", "sinks")}
    gu = convert_moe_packed_tensors(get(p + "mlp.experts.gate_up_proj_blocks"), get(p + "mlp.experts.gate_up_proj_scales"))
    dn = convert_moe_packed_tensors(get(p + "mlp.experts.down_proj_blocks"), get(p + "mlp.experts.down_proj_scales"))
    # the converter returns x @ W layouts: gate_up [E, H, 2I], down [E, I, H]; the card wants [N_out, K_in]
    assert tuple(gu.shape) == (NE, H, 2 * I) and tuple(dn.shape) == (NE, I, H), (gu.shape, dn.shape)
    gu_b, dn_b = get(p + "mlp.experts.gate_up_proj_bias"), get(p + "mlp.experts.down_proj_bias")
    common = {"ln_in": get(p + "input_layernorm.weight").to(torch.bfloat16),
              "ln_post": get(p + "post_attention_layernorm.weight").to(torch.bfloat16),
              "router_w": get(p + "mlp.router.weight").to(torch.bfloat16),
              "router_b": get(p + "mlp.router.bias").to(torch.bfloat16),
              "sliding": cfg["layer_types"][li] == "sliding_attention"}
    for k, c in enumerate(cards):
        qh, kvh = c["q_heads"], c["kv_heads"]
        q_rows = torch.cat([torch.arange(h * HD, (h + 1) * HD) for h in qh])
        kv_rows = torch.cat([torch.arange(h * HD, (h + 1) * HD) for h in kvh])
        n0, n1 = c["neurons"]
        ex = []
        for e in range(NE):
            g = gu[e][:, 0::2][:, n0:n1].t()          # gate rows of these neurons, [n, H]
            u = gu[e][:, 1::2][:, n0:n1].t()          # up rows, [n, H]
            ex.append({"gu": fp4(torch.cat([g, u], 0)),
                       "gu_b": torch.cat([gu_b[e][0::2][n0:n1], gu_b[e][1::2][n0:n1]]).to(torch.bfloat16),
                       "down": fp4(dn[e][n0:n1, :].t()),   # [H, n]
                       "down_b": dn_b[e].to(torch.bfloat16) if k == 0 else None})
        layer = dict(common)
        layer.update({
            "q": fp4(A["q_proj.weight"][q_rows]), "q_b": A["q_proj.bias"][q_rows].to(torch.bfloat16),
            "k": fp4(A["k_proj.weight"][kv_rows]), "k_b": A["k_proj.bias"][kv_rows].to(torch.bfloat16),
            "v": fp4(A["v_proj.weight"][kv_rows]), "v_b": A["v_proj.bias"][kv_rows].to(torch.bfloat16),
            "o": fp4(A["o_proj.weight"][:, q_rows]), "o_b": A["o_proj.bias"].to(torch.bfloat16) if k == 0 else None,
            "sinks": A["sinks"][qh].float(), "q_heads": qh, "kv_heads": kvh, "experts": ex})
        dst = os.path.join(OUT, "card_%02d" % k, "layer_%02d.pt" % li)
        torch.save(layer, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    print("LAYER|%d|%s|%.1f s" % (li, cfg["layer_types"][li], time.time() - t0), flush=True)

if DO_HEAD:
    t0 = time.time()
    head = get("lm_head.weight")
    for k, c in enumerate(cards):
        v0, v1 = c["vocab"]
        dst = os.path.join(OUT, "card_%02d" % k, "lm_head.pt")
        if not os.path.exists(dst):
            torch.save(fp4(head[v0:v1]), dst + ".tmp")
            os.replace(dst + ".tmp", dst)
    os.makedirs(WDIR, exist_ok=True)
    torch.save(get("model.norm.weight").to(torch.bfloat16), os.path.join(WDIR, "final_norm.pt"))
    e = get("model.embed_tokens.weight").to(torch.bfloat16).contiguous()
    with open(os.path.join(WDIR, "embed.bin.tmp"), "wb") as f:
        f.write(e.view(torch.int16).numpy().tobytes())
    os.replace(os.path.join(WDIR, "embed.bin.tmp"), os.path.join(WDIR, "embed.bin"))
    json.dump({"shape": list(e.shape)}, open(os.path.join(WDIR, "embed.json"), "w"))
    print("HEAD|%.1f s" % (time.time() - t0), flush=True)
print("DONE|%.0f s" % (time.time() - t_all), flush=True)
