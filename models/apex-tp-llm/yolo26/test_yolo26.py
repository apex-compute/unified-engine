"""Compile YOLO26-OBB for one engine and (optionally) run one image on it, unit by unit.

  python test_yolo26.py --compile-only
  python test_yolo26.py --dev xdma4 --engine 0 --image samples/boats.jpg --out /tmp/y26.jpg
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo26_card as Y                       # noqa: E402
U = Y.U

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26s-obb.pt"))
ap.add_argument("--size", type=int, default=640)
ap.add_argument("--compile-only", action="store_true")
ap.add_argument("--dev", default="xdma4")
ap.add_argument("--engine", type=int, default=0)
ap.add_argument("--pbase", type=lambda v: int(v, 0), default=0x10000000)
ap.add_argument("--psize", type=lambda v: int(v, 0), default=0x04000000)
ap.add_argument("--tbase", type=lambda v: int(v, 0), default=0x20000000)
ap.add_argument("--tsize", type=lambda v: int(v, 0), default=0x10000000)
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="/tmp/yolo26_out.jpg")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--repeat", type=int, default=2)
args = ap.parse_args()

ck = torch.load(args.weights, map_location="cpu", weights_only=False)
model = (ck.get("ema") or ck["model"]).float().eval()
names = model.names
win = args.engine * 0x80000000
t0 = time.time()
img, man = Y.compile_engine(model, args.size, args.size, pbase=win + args.pbase, plimit=win + args.pbase + args.psize,
                            tbase=win + args.tbase, tlimit=win + args.tbase + args.tsize)
print("compile %.1f s" % (time.time() - t0))
est = man["unit_est_s"]
print("units: %d, estimated longest %.1f ms (%s)" % (len(est), max(est) * 1e3, man["unit_names"][est.index(max(est))]))
if args.compile_only:
    sys.exit(0)

from yolov5_common import letterbox_image      # noqa: E402
from PIL import ImageDraw                      # noqa: E402

U.set_dma_device(args.dev)
U.configure_clock_from_hardware()
ue = U.UnifiedEngine(BASE_ADDR=U.UE_0_BASE_ADDR + args.engine * 0x10000,
                     conv_geometry_mode=U.CONV_GEOMETRY_QUEUE_CONFIG)
ue.software_reset(run_dram_self_test=False)
t = time.time()
ue.dma_write(ue.h2c_device, man["image_base"], torch.frombuffer(bytearray(img), dtype=torch.uint8), len(img))
print("image upload %.2f s (%.1f MB)" % (time.time() - t, len(img) / 1e6))

orig, chw, lb = letterbox_image(args.image, (args.size, args.size))
packed = Y.pack_input(chw)


def run_once():
    t = time.time()
    ue.dma_write(ue.h2c_device, man["input_address"], packed, man["input_bytes"])
    t_in = time.time() - t
    cyc = []
    for off in man["unit_offsets"]:
        ue.write_reg32(U.UE_INT_REG, 1)
        ue.start_execute_from_dram(man["program_address"] + off)
        dl = time.time() + 30
        while (ue.read_reg32(U.UE_INT_REG) & 3) != U.INT_CAUSE_HALT:
            if time.time() > dl:
                raise TimeoutError("unit at +%d did not halt" % off)
        cyc.append(ue.read_latency_cycles())
    t_run = time.time() - t - t_in
    bundle = torch.empty(man["head_bytes"] // 2, dtype=torch.bfloat16)
    ue.dma_read(ue.c2h_device, man["head_address"], bundle, man["head_bytes"])
    return bundle, cyc, t_in, t_run, time.time() - t


for r in range(args.repeat):
    bundle, cyc, t_in, t_run, t_all = run_once()
    ns = U.CLOCK_CYCLE_TIME_NS
    card = sum(cyc) * ns / 1e9
    print("run %d: card %.1f ms in %d units (longest %.2f ms), wall %.1f ms (input upload %.1f ms)"
          % (r, card * 1e3, len(cyc), max(cyc) * ns / 1e6, t_all * 1e3, t_in * 1e3))

if not torch.isfinite(bundle.float()).all():
    print("WARNING: non-finite values in the head output")
dets = Y.decode(man, bundle, conf=args.conf)
print("%d detections" % len(dets))
draw = ImageDraw.Draw(orig)
for cx, cy, w, h, a, s, c in dets:
    pts = [((x - lb.pad_left) / lb.ratio, (y - lb.pad_top) / lb.ratio) for x, y in Y.corners(cx, cy, w, h, a)]
    draw.polygon(pts, outline=(255, 60, 60), width=2)
    draw.text(pts[0], "%s %.2f" % (names[c], s), fill=(255, 255, 0))
    print("  %-16s %.2f  center (%.0f, %.0f)  %.0fx%.0f  angle %.2f" % (names[c], s, (cx - lb.pad_left) / lb.ratio,
                                                                       (cy - lb.pad_top) / lb.ratio, w / lb.ratio, h / lb.ratio, a))
orig.save(args.out)
units = [(c * U.CLOCK_CYCLE_TIME_NS / 1e6, n) for c, n in zip(cyc, man["unit_names"])]
json.dump({"units_ms": units, "est_ms": [e * 1e3 for e in man["unit_est_s"]]}, open(args.out + ".units.json", "w"))
print("saved", args.out)
