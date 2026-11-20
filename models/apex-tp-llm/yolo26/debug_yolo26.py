"""Debug tool (test only): run the YOLO26 card program op by op and compare every op's card output
with a float32 evaluation of the same lowered graph on the host. Not used by any serving path.

  chain SNR: against the float graph from the image (accumulated error)
  local SNR: the op alone, applied on the host to the card's own inputs (that op's error only)

  python debug_yolo26.py --dev xdma4 --engine 0
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

os.environ["YOLO_CUT_EACH_OP"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo26_card as Y                       # noqa: E402
from yolov5_common import letterbox_image     # noqa: E402
U = Y.U

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26s-obb.pt"))
ap.add_argument("--size", type=int, default=640)
ap.add_argument("--dev", default="xdma4")
ap.add_argument("--engine", type=int, default=0)
ap.add_argument("--image", required=True)
args = ap.parse_args()

ck = torch.load(args.weights, map_location="cpu", weights_only=False)
model = (ck.get("ema") or ck["model"]).float().eval()
win = args.engine * 0x80000000
img, man = Y.compile_engine(model, args.size, args.size, pbase=win + 0x08000000, plimit=win + 0x38000000,
                            tbase=win + 0x40000000, tlimit=win + 0x50000000)
g = Y.compile_engine.last_graph
ops = {op["name"]: op for op in g.ops}

_, chw, _ = letterbox_image(args.image, (args.size, args.size))
packed = Y.pack_input(chw)
c, h, w = chw.shape
x12 = chw.reshape(c, h // 2, 2, w // 2, 2).permute(2, 4, 0, 1, 3).reshape(4 * c, h // 2, w // 2)


def apply(op, ins):
    k = op["op"]
    if k == "conv":
        y = F.conv2d(ins[0][None], op["w"], op["b"], stride=op["stride"], padding=op["pad"])[0]
        return F.silu(y) if op["activate"] else y
    if k == "maxpool":
        return F.max_pool2d(ins[0][None], op["kernel"], op["stride"], op["pad"])[0]
    if k == "upsample2x":
        return F.interpolate(ins[0][None], scale_factor=2.0, mode="nearest")[0]
    if k == "add":
        return ins[0] + ins[1]
    if k == "concat":
        return torch.cat(ins, 0)
    q, kk, v = ins
    hd, hh, ww = v.shape
    n = hh * ww
    qf, kf, vf = q.reshape(-1, n).T, kk.reshape(-1, n).T, v.reshape(hd, n).T
    return (torch.softmax(qf @ kf.T, -1) @ vf).T.reshape(hd, hh, ww)


def snr(r, x):
    if not torch.isfinite(x).all():
        return -999.0
    sig, noi = float((r.double() ** 2).sum()), float(((x - r).double() ** 2).sum())
    return 10 * math.log10(max(sig, 1e-30) / max(noi, 1e-30))


inp = x12.to(torch.bfloat16).float()
ref = {"input": inp}
with torch.no_grad():
    for op in g.ops:
        ref[op["output"]] = apply(op, [ref[n] for n in op["inputs"]])

U.set_dma_device(args.dev)
U.configure_clock_from_hardware()
ue = U.UnifiedEngine(BASE_ADDR=U.UE_0_BASE_ADDR + args.engine * 0x10000,
                     conv_geometry_mode=U.CONV_GEOMETRY_QUEUE_CONFIG)
ue.software_reset(run_dram_self_test=False)
ue.dma_write(ue.h2c_device, man["image_base"], torch.frombuffer(bytearray(img), dtype=torch.uint8), len(img))
ue.dma_write(ue.h2c_device, man["input_address"], packed, man["input_bytes"])

by_unit = {}
for t in man["trace"]:
    by_unit.setdefault(t["unit"], []).append(t)
card = {"input": inp}
worst = []
for u, off in enumerate(man["unit_offsets"]):
    ue.write_reg32(U.UE_INT_REG, 1)
    ue.start_execute_from_dram(man["program_address"] + off)
    dl = time.time() + 30
    while (ue.read_reg32(U.UE_INT_REG) & 3) != U.INT_CAUSE_HALT:
        if time.time() > dl:
            raise TimeoutError("unit %d did not halt" % u)
    for t in by_unit.get(u, []):
        C, hh, ww = t["shape"]
        P = t["physical"]
        n = hh * ww
        buf = torch.empty(n * P, dtype=torch.bfloat16)
        ue.dma_read(ue.c2h_device, t["address"], buf, n * P * 2)
        x = buf.float().view(n, P)[:, torch.tensor(t["channel_map"])].T.reshape(C, hh, ww)
        card[t["name"]] = x
        op = ops[t["name"]]
        with torch.no_grad():
            loc = apply(op, [card[i] for i in op["inputs"]])
        s_chain, s_loc = snr(ref[t["name"]], x), snr(loc, x)
        worst.append((s_loc, t["name"]))
        print("%-26s %-10s %-16s chain %6.1f dB   local %6.1f dB%s"
              % (t["name"], t["op"], str(tuple(t["shape"])), s_chain, s_loc, "   <== LOW" if s_loc < 18 else ""))
worst.sort()
print("lowest local SNR:", ", ".join("%s %.1f" % (n, s) for s, n in worst[:12]))
