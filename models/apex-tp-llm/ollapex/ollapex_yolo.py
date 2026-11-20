"""OllApeX YOLO portal: YOLO26-OBB (DOTA classes) on the same cards as the LM, next to it in each engine's free
memory. The server only resizes / packs a frame and decodes the result; the network runs on the cards.

  GET  /yolo                 the portal page
  GET  /api/yolo/status      engines holding YOLO, per-engine frames and time
  POST /api/yolo/detect      raw image body (X-Mode: camera | image, X-Conf) -> detections (waits for the cards)
  POST /api/yolo/video       raw video body (X-Filename, X-Fps, X-Conf) -> {job}; frames run before the next chat
  GET  /api/yolo/job?id=     progress, per-frame detections with track ids, unique objects per class
"""
import glob
import io
import itertools
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse

import numpy as np

YOLO_CODE = os.path.expanduser(os.environ.get("YOLO_CODE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolo26")))
WORK = "/dev/shm/ollapex_yolo"
MAX_VIDEO = 1 << 30
MAX_FRAMES = 600
BUS = YB = None
LOCK = threading.Lock()
RESERVED = set()
SEQ = itertools.count(1)
JOBS = {}
ENGINE = None


def init(engine_getter):
    """Before the driver starts: create the frame bus; the driver and the card processes inherit its name."""
    global BUS, YB, ENGINE
    ENGINE = engine_getter
    try:
        if YOLO_CODE not in sys.path:
            sys.path.insert(0, YOLO_CODE)
        import yolo26_bus
        YB = yolo26_bus
        BUS = YB.Bus()
        os.environ["YOLO_BUS"] = json.dumps(BUS.names)
        os.makedirs(WORK, exist_ok=True)
    except Exception as e:
        print("YOLO portal disabled: %s" % e, flush=True)
        BUS = None


def engines():
    st = ENGINE().status
    info = st.get("info") or {}
    n = int(info.get("yolo_engines") or 0) if st.get("state") in ("ready", "busy") else 0
    return n, st


def status():
    n, st = engines()
    out = {"available": bool(BUS is not None and n), "engines": n, "model": st.get("model"), "state": st.get("state"),
           "size": YB.SIZE if YB else None, "classes": YB.NAMES if YB else []}
    if BUS is not None:
        k = [i for i in range(YB.P_MAX) if BUS.eng_ready[i]]
        out["per_engine"] = [{"engine": i, "frames": int(BUS.eng_frames[i]), "last_ms": round(float(BUS.eng_ms[i]), 1),
                              "busy": bool(BUS.eng_slot[i] >= 0)} for i in k]
        out["queued"] = int((BUS.slot_state == YB.QUEUED).sum())
        out["running"] = int((BUS.slot_state == YB.ASSIGNED).sum())
        cards = [c for c in ((st.get("info") or {}).get("cards") or [])]
        y = [c.get("yolo") for c in cards if c.get("yolo")]
        if y:
            ok = [v for v in y if "error" not in v]
            out["build"] = {"image_mb": round(max((v["image_mb"] for v in ok), default=0), 1),
                            "tensor_mb": round(max((v["tensor_mb"] for v in ok), default=0), 1),
                            "units": max((v["units"] for v in ok), default=0),
                            "precision": ok[0].get("precision") if ok else None,
                            "errors": [v["error"] for v in y if "error" in v]}
    return out


ORPHANS = set()                                   # slots whose waiter gave up; free once the cards are done


def _slot(timeout):
    t_end = time.time() + timeout
    while True:
        with LOCK:
            for s in list(ORPHANS):
                if BUS.slot_state[s] in (YB.DONE, YB.FAILED):
                    BUS.slot_state[s] = YB.FREE
                    ORPHANS.discard(s)
            for s in range(YB.NSLOT):
                if s in ORPHANS:
                    continue
                if BUS.slot_state[s] == YB.FREE and s not in RESERVED:
                    RESERVED.add(s)
                    return s
        if time.time() > t_end:
            raise RuntimeError("all frame slots are busy")
        time.sleep(0.005)


def run_frame(img, mode, conf=0.25, timeout=120.0):
    """PIL image -> detections in its own pixels, plus timings. Blocks until the cards have run it."""
    n, _ = engines()
    if BUS is None or not n:
        raise ValueError("YOLO needs Gemma 4 26B loaded on the cards (load it on the chat page).")
    t0 = time.time()
    rgb, r, left, top = YB.letterbox(img)
    s = _slot(timeout)
    try:
        YB.pack(rgb, BUS.inp[s])
        t_pack = time.time() - t0
        BUS.slot_mode[s] = YB.CAMERA if mode == "camera" else YB.UPLOAD
        BUS.slot_seq[s] = next(SEQ)
        BUS.slot_t[s, 0] = time.time()
        BUS.slot_engine[s] = -1
        BUS.slot_state[s] = YB.QUEUED
        with LOCK:
            RESERVED.discard(s)
        t_end = time.time() + timeout
        while BUS.slot_state[s] in (YB.QUEUED, YB.ASSIGNED):
            if time.time() > t_end:
                raise RuntimeError("the cards did not finish the frame in %.0f s" % timeout)
            time.sleep(0.003)
        if BUS.slot_state[s] != YB.DONE:
            raise RuntimeError("the frame failed on the cards (model reloaded?)")
        out = BUS.out[s].copy()
        eng, t = int(BUS.slot_engine[s]), BUS.slot_t[s].copy()
    finally:
        with LOCK:
            RESERVED.discard(s)
            if BUS.slot_state[s] in (YB.QUEUED, YB.ASSIGNED):
                ORPHANS.add(s)
            else:
                BUS.slot_state[s] = YB.FREE
    t1 = time.time()
    dets = YB.detections_json(YB.decode(out, conf), r, left, top, img.width, img.height)
    return {"detections": dets, "width": img.width, "height": img.height, "engine": eng,
            "timing": {"pack_ms": round(t_pack * 1e3, 1), "queue_ms": round((t[1] - t[0]) * 1e3, 1),
                       "card_ms": round(float(t[3]), 1), "decode_ms": round((time.time() - t1) * 1e3, 1),
                       "total_ms": round((time.time() - t0) * 1e3, 1)}}


def counts(dets):
    c = {}
    for d in dets:
        c[d["cls"]] = c.get(d["cls"], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def ffmpeg():
    hits = glob.glob(os.path.join(YOLO_CODE, "env", "lib", "python3*", "site-packages", "imageio_ffmpeg",
                                  "binaries", "ffmpeg-*"))
    return hits[0] if hits else "ffmpeg"


def video_job(jid, path, fps, conf):
    from PIL import Image
    job = JOBS[jid]
    try:
        fdir = path + ".frames"
        os.makedirs(fdir, exist_ok=True)
        job["state"] = "extracting"
        r = subprocess.run([ffmpeg(), "-loglevel", "error", "-i", path, "-vf", "fps=%g" % fps, "-frames:v",
                            str(MAX_FRAMES), "-q:v", "2", os.path.join(fdir, "f%05d.jpg")],
                           capture_output=True, text=True, timeout=600)
        files = sorted(glob.glob(os.path.join(fdir, "f*.jpg")))
        if not files:
            raise RuntimeError("no frames could be read from the video: %s" % (r.stderr or "")[-300:])
        job.update(state="running", total=len(files))
        results = [None] * len(files)
        lock = threading.Lock()
        nxt = iter(range(len(files)))

        def one():
            while True:
                with lock:
                    i = next(nxt, None)
                if i is None:
                    return
                with Image.open(files[i]) as im:
                    res = run_frame(im.convert("RGB"), "upload", conf, timeout=600)
                res["t"] = i / fps
                results[i] = res
                with lock:
                    job["done"] += 1

        n, _ = engines()
        th = [threading.Thread(target=one, daemon=True) for _ in range(max(1, min(n + 2, YB.NSLOT)))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if any(x is None for x in results):
            raise RuntimeError("some frames failed")
        tr = YB.Tracker()
        uniq = {}
        for res in results:
            tr.update(res["detections"])
            for d in res["detections"]:
                uniq.setdefault(d["cls"], set()).add(d["track"])
        job.update(frames=results, unique={k: len(v) for k, v in sorted(uniq.items(), key=lambda kv: -len(kv[1]))},
                   peak={k: max(counts(f["detections"]).get(k, 0) for f in results) for k in uniq},
                   card_ms=round(float(np.mean([f["timing"]["card_ms"] for f in results])), 1),
                   wall_s=round(time.time() - job["t0"], 2), state="done")
    except Exception as e:
        job.update(state="error", error=str(e))
    finally:
        try:
            for f in glob.glob(path + ".frames/*"):
                os.remove(f)
            os.rmdir(path + ".frames")
            os.remove(path)
        except OSError:
            pass


def handle_get(h, path):
    if path == "/api/yolo/status":
        return h.send_json(status())
    if path == "/api/yolo/job":
        q = urllib.parse.parse_qs(urllib.parse.urlparse(h.path).query)
        job = JOBS.get((q.get("id") or [""])[0])
        if not job:
            return h.send_json({"error": "no such job"}, 404)
        return h.send_json({k: v for k, v in job.items() if k != "t0"})
    return h.send_json({"error": "not found"}, 404)


def handle_post(h, path):
    n = int(h.headers.get("Content-Length") or 0)
    conf = float(h.headers.get("X-Conf") or 0.25)
    if path == "/api/yolo/detect":
        from PIL import Image, ImageOps
        if n <= 0 or n > 32 << 20:
            raise ValueError("send one image (at most 32 MB)")
        data = h.rfile.read(n)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        res = run_frame(img, (h.headers.get("X-Mode") or "image").lower(), conf)
        res["counts"] = counts(res["detections"])
        return h.send_json(res)
    if path == "/api/yolo/video":
        if n <= 0 or n > MAX_VIDEO:
            raise ValueError("send one video (at most %d MB)" % (MAX_VIDEO >> 20))
        if not engines()[0]:
            raise ValueError("YOLO needs Gemma 4 26B loaded on the cards (load it on the chat page).")
        fps = max(0.1, min(float(h.headers.get("X-Fps") or 1.0), 10.0))
        jid = "%d%04d" % (int(time.time()), next(SEQ) % 10000)
        name = os.path.basename(urllib.parse.unquote(h.headers.get("X-Filename") or "video.mp4"))
        path = os.path.join(WORK, jid + "_" + name)
        left = n
        with open(path, "wb") as f:
            while left:
                b = h.rfile.read(min(left, 1 << 20))
                if not b:
                    raise ValueError("upload interrupted")
                f.write(b)
                left -= len(b)
        JOBS[jid] = {"id": jid, "name": name, "fps": fps, "state": "queued", "done": 0, "total": 0, "t0": time.time()}
        for old in sorted(JOBS)[:-20]:
            JOBS.pop(old, None)
        threading.Thread(target=video_job, args=(jid, path, fps, conf), daemon=True).start()
        return h.send_json({"job": jid})
    return h.send_json({"error": "not found"}, 404)
