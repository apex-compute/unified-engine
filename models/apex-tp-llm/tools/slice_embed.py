"""Split the (sqrt(hidden)-scaled, bf16) embedding table over the shards, along the same 64-row vocabulary
blocks as the LM head: card_XX/embed.pt holds rows vocab[0]..vocab[1]-1 of shard XX. The cards keep the
table; the host only copies the one row a token needs from the card that holds it (host-side preparation)."""
import json
import os

import numpy as np
import torch

WORK = os.path.expanduser(os.environ.get("EMB_WORK", os.path.join(os.environ.get("APEX_TP_WORK", "~/apex_tp"), "models", "gemma4_26b")))
TP = os.path.join(WORK, os.environ.get("TPDIR", "tp10"))
WDIR = os.path.join(WORK, "weights")
plan = json.load(open(os.path.join(TP, "plan.json")))
shape = json.load(open(os.path.join(WDIR, "embed.json")))["shape"]
emb = np.memmap(os.path.join(WDIR, "embed.bin"), dtype=np.int16, mode="r", shape=tuple(shape))
for k, c in enumerate(plan["cards"]):
    v0, v1 = c["vocab"]
    dst = os.path.join(TP, "card_%02d" % k, "embed.pt")
    t = torch.from_numpy(np.array(emb[v0:v1])).view(torch.bfloat16).contiguous()
    torch.save(t, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    print("shard %d rows %d..%d -> %s (%.0f MB)" % (k, v0, v1, dst, t.numel() * 2 / 1e6), flush=True)
