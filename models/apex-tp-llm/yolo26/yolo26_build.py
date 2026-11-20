"""Build YOLO26-OBB for one engine's free DRAM ranges: <out>.img (parameters + program) and <out>.json.
Run by yolo26_bus.build() in yolo26/env (torch + ultralytics + the vendor compiler)."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo26_card as Y      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--size", type=int, default=640)
for k in ("pbase", "plimit", "tbase", "tlimit"):
    ap.add_argument("--" + k, type=lambda v: int(v, 0), required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ck = torch.load(a.weights, map_location="cpu", weights_only=False)
model = (ck.get("ema") or ck["model"]).float().eval()
img, man = Y.compile_engine(model, a.size, a.size, pbase=a.pbase, plimit=a.plimit, tbase=a.tbase, tlimit=a.tlimit,
                            log=lambda *x: print(*x, file=sys.stderr))
man.pop("trace", None)
man.update(names=[model.names[i] for i in range(len(model.names))], precision=Y.PRECISION,
           stream_kb=Y.STREAM_BUDGET // 1024)
with open(a.out + ".img.tmp", "wb") as f:
    f.write(img)
with open(a.out + ".json.tmp", "w") as f:
    json.dump(man, f)
os.replace(a.out + ".img.tmp", a.out + ".img")
os.replace(a.out + ".json.tmp", a.out + ".json")
