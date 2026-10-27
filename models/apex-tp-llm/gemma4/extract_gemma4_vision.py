"""Gemma 4 vision tower in the card layout (host-side preparation, once) -> $GEMMA4_WORK/vision/vision.pt

Precision: IF8 for the 27 encoder layers (FP4 measured 27 % mean token error, IF8 2 %), bf16 for the patch
projection, the projection into the language model and every table.

Head layout: head_dim 72 is padded to 128. Within a head the real dims are placed so that the card's
rotate-half RoPE (pairs (j, j + 64)) does exactly transformers' axial RoPE (two 36-wide parts, x then y,
each rotated as pairs (i, i + 18)):
    padded j:   0..17  -> x dims 0..17      64..81  -> x dims 18..35
               18..35  -> y dims 36..53     82..99  -> y dims 54..71      the rest stays 0
q, k, v rows and the o-projection columns follow it; q_norm / k_norm / v_norm gammas are multiplied by
sqrt(72 / 128) so an RMS over the 128 padded dims equals the model's RMS over its 72.

Tables (positions 0..POS-1 of each axis, B layout [N, K] with K = 2 * POS: x positions, then y positions):
    pos_t   [1152, 2*POS]  x / y position embeddings          (one-hot(x, y) @ pos_t^T  = patch position)
    rope_t  [256,  2*POS]  [cos | sinS] of the padded head     (+ rope_bias: cos = 1 on the padding dims)
with sinS = [-sin, +sin] over the two 64-halves (the card computes x*cos + swap_halves(x)*sinS).
"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/unified-engine"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_lib as Q  # noqa: E402
from st_read import SafeDir  # noqa: E402

MODEL = os.path.expanduser(os.environ.get("GEMMA4_MODEL", "~/apex_exp/models/gemma4_26b/hf/gemma-4-26b-a4b-it"))
WORK = os.path.expanduser(os.environ.get("GEMMA4_WORK", os.path.join(os.environ.get("APEX_TP_WORK", "~/apex_tp"),
                                                                       "models", "gemma4_26b")))
OUT = os.path.join(WORK, "vision", "vision.pt")
POS = int(os.environ.get("VISION_POS", "1024"))        # largest patch index per axis the tables cover
torch.set_num_threads(int(os.environ.get("THREADS", "2")))

cfg = json.load(open(os.path.join(MODEL, "config.json")))
vc = cfg["vision_config"]
HID, NH, HD, I = vc["hidden_size"], vc["num_attention_heads"], vc["head_dim"], vc["intermediate_size"]
HDP, IP = 128, (I + 63) // 64 * 64
assert HD == 72 and HID % 64 == 0
S = SafeDir(MODEL)
P = "model.vision_tower."


def get(k):
    return S.get(k).float()


def if8(w):
    """[N, K] -> (scales bf16, data uint8) on the card's IF8 (one scale per 64 along K)."""
    assert w.dim() == 2 and w.shape[1] % 64 == 0, w.shape
    d, s = Q.quantize_if8(w.to(torch.bfloat16).contiguous(), 64)
    return (torch.frombuffer(bytearray(s), dtype=torch.bfloat16).clone(),
            torch.frombuffer(bytearray(d), dtype=torch.uint8).clone())


# real head dim -> padded position
PERM = torch.cat([torch.arange(0, 18), torch.arange(64, 82), torch.arange(18, 36), torch.arange(82, 100)])
NC = math.sqrt(HD / HDP)                                  # RMS over 128 padded dims -> over the 72 real ones


def pad_rows(w):
    """[NH*HD, K] -> [NH*HDP, K]: each head's rows to their padded places, zeros elsewhere."""
    out = torch.zeros(NH * HDP, w.shape[1])
    for h in range(NH):
        out[h * HDP + PERM] = w[h * HD:(h + 1) * HD]
    return out


def pad_cols(w):
    """[N, NH*HD] -> [N, NH*HDP]."""
    out = torch.zeros(w.shape[0], NH * HDP)
    for h in range(NH):
        out[:, h * HDP + PERM] = w[:, h * HD:(h + 1) * HD]
    return out


def pad_gamma(g):
    out = torch.zeros(HDP)
    out[PERM] = g * NC
    return out


t0 = time.time()
layers = []
for li in range(vc["num_hidden_layers"]):
    p = P + "encoder.layers.%d." % li
    a = p + "self_attn."
    gate = torch.zeros(IP, HID); gate[:I] = get(p + "mlp.gate_proj.linear.weight")
    up = torch.zeros(IP, HID); up[:I] = get(p + "mlp.up_proj.linear.weight")
    down = torch.zeros(HID, IP); down[:, :I] = get(p + "mlp.down_proj.linear.weight")
    layers.append({
        "q": if8(pad_rows(get(a + "q_proj.linear.weight"))), "k": if8(pad_rows(get(a + "k_proj.linear.weight"))),
        "v": if8(pad_rows(get(a + "v_proj.linear.weight"))), "o": if8(pad_cols(get(a + "o_proj.linear.weight"))),
        "q_norm": pad_gamma(get(a + "q_norm.weight")).to(torch.bfloat16),
        "k_norm": pad_gamma(get(a + "k_norm.weight")).to(torch.bfloat16),
        "gate": if8(gate), "up": if8(up), "down": if8(down),
        "ln_in": get(p + "input_layernorm.weight").to(torch.bfloat16),
        "ln_post_attn": get(p + "post_attention_layernorm.weight").to(torch.bfloat16),
        "ln_pre_ff": get(p + "pre_feedforward_layernorm.weight").to(torch.bfloat16),
        "ln_post_ff": get(p + "post_feedforward_layernorm.weight").to(torch.bfloat16),
    })
    print("LAYER|%d|%.0f s" % (li, time.time() - t0), flush=True)

v_norm = torch.zeros(HDP)
v_norm[PERM] = NC                                         # v_norm has no weight: only the 72/128 correction

pe = get(P + "patch_embedder.position_embedding_table")  # [2, 10240, HID]
pos_t = torch.cat([pe[0, :POS].t(), pe[1, :POS].t()], 1)  # [HID, 2*POS]

theta = vc["rope_parameters"]["rope_theta"]
inv = 1.0 / (theta ** (torch.arange(0, HD // 2, 2, dtype=torch.float32) / (HD // 2)))   # 18 frequencies
pos = torch.arange(POS, dtype=torch.float32)
rope_t = torch.zeros(2 * HDP, 2 * POS)                    # rows: cos (128) then sinS (128); cols: x then y positions
for axis, (lo, hi) in enumerate(((0, 64), (18, 82))):     # x dims sit at 0..17 / 64..81, y at 18..35 / 82..99
    ang = torch.outer(inv, pos)                           # [18, POS]
    c0 = axis * POS
    rope_t[lo:lo + 18, c0:c0 + POS] = ang.cos()
    rope_t[hi:hi + 18, c0:c0 + POS] = ang.cos()
    rope_t[HDP + lo:HDP + lo + 18, c0:c0 + POS] = -ang.sin()
    rope_t[HDP + hi:HDP + hi + 18, c0:c0 + POS] = ang.sin()
rope_bias = torch.zeros(2 * HDP)
real = torch.zeros(HDP, dtype=torch.bool); real[PERM] = True
rope_bias[:HDP][~real] = 1.0                              # cos = 1, sin = 0 on the padding dims

pool = vc.get("pooling_kernel_size", 3)
std_scale, std_bias = get(P + "std_scale"), get(P + "std_bias")
out = {
    "dims": {"HID": HID, "NH": NH, "HD": HD, "HDP": HDP, "I": I, "IP": IP, "POS": POS, "POOL": pool,
             "PATCH": vc["patch_size"], "DEPTH": vc["num_hidden_layers"], "EPS": vc["rms_norm_eps"],
             "OUT": cfg["text_config"]["hidden_size"], "soft_tokens": vc.get("default_output_length", 280)},
    "patch_w": get(P + "patch_embedder.input_proj.weight").to(torch.bfloat16),       # [HID, 768] (pixels as 2x-1)
    "pos_t": pos_t.to(torch.bfloat16), "rope_t": rope_t.to(torch.bfloat16), "rope_bias": rope_bias.to(torch.bfloat16),
    "v_norm": v_norm.to(torch.bfloat16), "layers": layers,
    # pooled mean * sqrt(HID), then (x - std_bias) * std_scale, as ONE matmul: x_sum @ diag(d) + c
    "std_diag": torch.diag(std_scale * math.sqrt(HID) / (pool * pool)).to(torch.bfloat16),
    "std_c": (-std_bias * std_scale).to(torch.bfloat16),
    "embed_w": get("model.embed_vision.embedding_projection.weight").to(torch.bfloat16),  # [OUT, HID]
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(out, OUT + ".tmp")
os.replace(OUT + ".tmp", OUT)
print("DONE|%s|%.0f MB|%.0f s" % (OUT, os.path.getsize(OUT) / 1e6, time.time() - t0), flush=True)
