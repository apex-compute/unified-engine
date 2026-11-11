"""Like slice_embed.py, but the shard's embedding rows are stored as IF8 (one bf16 scale per 64 values; each
block INT8 or FP8 E4M3, whichever reconstructs it better) for models whose bf16 table does not fit next to
the weights. card_XX/embed8.pt = {"data": uint8 [rows*H], "scales": bf16 [rows*H/64], "rows": n, "H": H}.
The card dequantizes a token's row itself (UE_MODE.DEQUANTIZE). Also prints the reconstruction error."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/unified-engine"))
import quant_lib as Q  # noqa: E402

WORK = os.path.expanduser(os.environ.get("EMB_WORK", os.path.join(os.environ.get("APEX_TP_WORK", "~/apex_tp"), "models", "q38_27b")))
TP = os.path.join(WORK, os.environ.get("TPDIR", "tp10fp4"))
WDIR = os.path.join(WORK, "weights")
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
plan = json.load(open(os.path.join(TP, "plan.json")))
shape = json.load(open(os.path.join(WDIR, "embed.json")))["shape"]
H = shape[1]
emb = np.memmap(os.path.join(WDIR, "embed.bin"), dtype=np.int16, mode="r", shape=tuple(shape))
for k, c in enumerate(plan["cards"]):
    v0, v1 = c["vocab"]
    w = torch.from_numpy(np.array(emb[v0:v1])).view(torch.bfloat16)
    d, s = Q.quantize_if8(w, 64)
    data = torch.frombuffer(bytearray(d), dtype=torch.uint8).clone()
    scales = torch.frombuffer(bytearray(s), dtype=torch.bfloat16).clone()
    rec = Q.dequantize_if8(d, s, v1 - v0, H, 64).float()
    err = ((rec - w.float()).norm(dim=1) / w.float().norm(dim=1).clamp(min=1e-9))
    dst = os.path.join(TP, "card_%02d" % k, "embed8.pt")
    torch.save({"data": data, "scales": scales, "rows": v1 - v0, "H": H}, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print("shard %d rows %d..%d -> %s (%.0f MB) | row rel err mean %.4f max %.4f" % (
        k, v0, v1, dst, (data.numel() + scales.numel() * 2) / 1e6, float(err.mean()), float(err.max())), flush=True)
