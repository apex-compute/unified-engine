"""YOLO26-OBB next to an LLM on the cards: the shared frame bus, the per-engine build (cached), and the host's
management work around a frame (letterbox + packing into the card's input layout, decoding the head maps).

Bus (shared memory; the server, the driver and the card processes all attach):
  slot_state  0 free, 1 queued, 2 assigned to an engine, 3 done (heads in out[slot]), 4 failed
  slot_mode   1 camera (runs in slices next to the LLM), 2 upload (runs before the next chat request)
  eng_slot    per engine: the slot it works on (-1: none);  eng_ready: 1 once the engine holds YOLO
The server fills a slot and queues it, the driver hands queued slots to free engines and gives the card
processes time (SEG_YOLO with a budget), a card process loads a slot's frame onto its engine, runs program
units until the budget is used, and on the last unit copies the head maps back and marks the slot done.
"""

import hashlib
import json
import os
import subprocess
import time

import numpy as np
from multiprocessing import shared_memory

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = int(os.environ.get("YOLO_SIZE", "640"))
STRIDES = (8, 16, 32)
NC = 15
NAMES = ["plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
         "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
         "roundabout", "soccer ball field", "swimming pool"]
LANES = 64
IN_ELEMS = (SIZE // 2) * (SIZE // 2) * LANES                       # 2x2 space-to-depth, one 64-lane line per pixel
HEADS = [((SIZE // s) ** 2) * LANES for s in STRIDES]              # each level: [pixels, 64 lanes] bf16
OUT_ELEMS = sum(HEADS)
NSLOT = int(os.environ.get("YOLO_SLOTS", "16"))
P_MAX = 16
FREE, QUEUED, ASSIGNED, DONE, FAILED = 0, 1, 2, 3, 4
CAMERA, UPLOAD = 1, 2

SPEC = (("slot_state", np.int64, (NSLOT,)), ("slot_mode", np.int64, (NSLOT,)), ("slot_seq", np.int64, (NSLOT,)),
        ("slot_engine", np.int64, (NSLOT,)), ("slot_t", np.float64, (NSLOT, 4)),   # queued, assigned, done, card ms
        ("eng_slot", np.int64, (P_MAX,)), ("eng_ready", np.int64, (P_MAX,)),
        ("eng_frames", np.int64, (P_MAX,)), ("eng_ms", np.float64, (P_MAX,)),
        ("inp", np.int16, (NSLOT, IN_ELEMS)), ("out", np.int16, (NSLOT, OUT_ELEMS)))


class Bus:
    def __init__(self, names=None):
        self.blocks, self.names = [], []
        for i, (name, dtype, shape) in enumerate(SPEC):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                try:
                    shm = shared_memory.SharedMemory(name=names[i], track=False)
                except TypeError:
                    shm = shared_memory.SharedMemory(name=names[i])
            self.blocks.append(shm)
            self.names.append(shm.name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            if names is None:
                arr[...] = 0
            setattr(self, name, arr)
        if names is None:
            self.eng_slot[:] = -1
            self.slot_engine[:] = -1

    @staticmethod
    def from_env():
        v = os.environ.get("YOLO_BUS")
        return Bus(json.loads(v)) if v else None

    def pending(self, modes=(CAMERA, UPLOAD)):
        s = self.slot_state
        return bool(((s == QUEUED) | (s == ASSIGNED))[np.isin(self.slot_mode, modes)].any())

    def close(self, unlink=False):
        for b in self.blocks:
            try:
                b.close()
                if unlink:
                    b.unlink()
            except Exception:
                pass


# ----------------------------------------------------------------------------------------------
# per-engine build, in a separate interpreter (yolo26/env: torch + ultralytics + the vendor compiler)

def build(pbase, plimit, tbase, tlimit, log=print):
    """Image + manifest for one engine's free ranges, from the cache when this layout was built before."""
    py = os.path.expanduser(os.environ.get("YOLO_PY", os.path.join(HERE, "env", "bin", "python")))
    weights = os.path.expanduser(os.environ.get("YOLO_WEIGHTS", os.path.join(HERE, "yolo26s-obb.pt")))
    prec, kb = os.environ.get("YOLO_PREC", "if8"), os.environ.get("YOLO_STREAM_KB", "256")
    h = hashlib.sha1()
    for f in (weights, os.path.join(HERE, "yolo26_card.py"), os.path.join(HERE, "yolo26_build.py")):
        st = os.stat(f)
        h.update(("%s %d %d;" % (f, st.st_size, int(st.st_mtime))).encode())
    h.update(("%d %x %x %x %x %s %s" % (SIZE, pbase, plimit, tbase, tlimit, prec, kb)).encode())
    cdir = os.path.join(HERE, "cache")
    os.makedirs(cdir, exist_ok=True)
    pre = os.path.join(cdir, h.hexdigest()[:20])
    t0 = time.time()
    cached = os.path.exists(pre + ".json") and os.path.exists(pre + ".img")
    if not cached:
        env = dict(os.environ, YOLO_PREC=prec, YOLO_STREAM_KB=kb)
        r = subprocess.run([py, os.path.join(HERE, "yolo26_build.py"), "--weights", weights, "--size", str(SIZE),
                            "--pbase", hex(pbase), "--plimit", hex(plimit), "--tbase", hex(tbase),
                            "--tlimit", hex(tlimit), "--out", pre], env=env, capture_output=True, text=True,
                           timeout=900, preexec_fn=lambda: os.sched_setaffinity(0, range(os.cpu_count())))
        if r.returncode != 0:
            raise RuntimeError("YOLO build failed: %s" % (r.stderr or r.stdout)[-1500:])
    man = json.load(open(pre + ".json"))
    img = open(pre + ".img", "rb").read()
    man.update(cached=cached, build_s=time.time() - t0)
    return img, man


# ----------------------------------------------------------------------------------------------
# host side of a frame: letterbox + packing, and decoding the head maps

def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).view(np.int16)


def letterbox(img):
    """PIL image -> (uint8 [SIZE, SIZE, 3], ratio, pad_left, pad_top); grey 114 padding as ultralytics."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    r = min(SIZE / h, SIZE / w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    left, top = round((SIZE - nw) / 2 - 0.1), round((SIZE - nh) / 2 - 0.1)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    return np.asarray(canvas), r, left, top


def pack(rgb, out):
    """uint8 [SIZE, SIZE, 3] -> out (int16 view of bf16, IN_ELEMS): 2x2 space-to-depth, channel a*6 + b*3 + c,
    lanes 12..63 zero (yolo26_card.stem_s2d)."""
    h2 = SIZE // 2
    x = rgb.reshape(h2, 2, h2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(h2 * h2, 12).astype(np.float32) / 255.0
    o = out.reshape(h2 * h2, LANES)
    o[:, 12:] = 0
    o[:, :12] = bf16_bits(x)


def decode(out, conf=0.25, max_det=300):
    """Head maps of one frame -> [(cx, cy, w, h, angle, score, cls)] in letterboxed input pixels."""
    rows, anc, st = [], [], []
    off = 0
    for s, n in zip(STRIDES, HEADS):
        g = SIZE // s
        a = out[off:off + n].view(np.int16).astype(np.int32) << 16
        rows.append(a.view(np.float32).reshape(g * g, LANES)[:, :5 + NC])
        ys, xs = np.mgrid[0:g, 0:g]
        anc.append(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1))
        st.append(np.full((g * g, 1), float(s)))
        off += n
    p, anc, st = np.concatenate(rows), np.concatenate(anc), np.concatenate(st)
    lt, rb, logits, ang = p[:, 0:2], p[:, 2:4], p[:, 4:4 + NC], p[:, 4 + NC:5 + NC]
    cos, sin = np.cos(ang), np.sin(ang)
    xf, yf = np.split((rb - lt) / 2, 2, 1)
    xy = np.concatenate([xf * cos - yf * sin, xf * sin + yf * cos], 1) + anc
    box = np.concatenate([xy, lt + rb], 1) * st
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    k = min(max_det, len(scores))
    best = np.argpartition(-scores.max(1), k - 1)[:k]
    flat = scores[best].ravel()
    order = np.argsort(-flat)[:k]
    res = []
    for o in order:
        v = float(flat[o])
        if v < conf:
            break
        i, c = int(best[o // NC]), int(o % NC)
        res.append((*map(float, box[i]), float(ang[i, 0]), v, c))
    return res


def polygon(cx, cy, w, h, a):
    c, s = np.cos(a), np.sin(a)
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def detections_json(dets, r, left, top, width, height):
    """Letterboxed detections -> original-image coordinates, as the web page draws them."""
    out = []
    for cx, cy, w, h, a, sc, c in dets:
        pts = [((x - left) / r, (y - top) / r) for x, y in polygon(cx, cy, w, h, a)]
        out.append({"cls": NAMES[c], "score": round(sc, 3), "angle": round(a, 3),
                    "cx": round((cx - left) / r, 1), "cy": round((cy - top) / r, 1),
                    "w": round(w / r, 1), "h": round(h / r, 1),
                    "poly": [[round(x, 1), round(y, 1)] for x, y in pts]})
    return out


class Tracker:
    """Greedy IoU tracker on the boxes' axis-aligned extents (enough for slow ships / parked planes).
    A track survives `keep` frames without a match; every track gets a stable id."""

    def __init__(self, iou=0.3, keep=3):
        self.iou, self.keep = iou, keep
        self.tracks = {}            # id -> {"box", "cls", "miss", "hits"}
        self.next = 1

    @staticmethod
    def _aabb(d):
        xs, ys = [p[0] for p in d["poly"]], [p[1] for p in d["poly"]]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0.0

    def update(self, dets):
        boxes = [self._aabb(d) for d in dets]
        pairs = sorted(((self._iou(t["box"], b), tid, j) for tid, t in self.tracks.items()
                        for j, b in enumerate(boxes) if t["cls"] == dets[j]["cls"]), reverse=True)
        used_t, used_d = set(), set()
        for v, tid, j in pairs:
            if v < self.iou or tid in used_t or j in used_d:
                continue
            used_t.add(tid)
            used_d.add(j)
            self.tracks[tid].update(box=boxes[j], miss=0, hits=self.tracks[tid]["hits"] + 1)
            dets[j]["track"] = tid
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.keep:
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks[self.next] = {"box": boxes[j], "cls": d["cls"], "miss": 0, "hits": 1}
                d["track"] = self.next
                self.next += 1
        return dets
