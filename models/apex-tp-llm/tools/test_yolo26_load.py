"""Keep N frames in flight through /api/yolo/detect (camera mode) for D seconds; report frames per second."""
import json, sys, threading, time, urllib.request
N, D = int(sys.argv[1]), float(sys.argv[2])
img = open(sys.argv[3], "rb").read()
done, lat, errs, card = [], [], [], []
t_end = time.time() + D
def one():
    while time.time() < t_end:
        t = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:3000/api/yolo/detect", data=img,
                headers={"X-Mode": "camera", "X-Conf": "0.25"}), timeout=300)
            d = json.loads(r.read())
            done.append(time.time()); lat.append(time.time() - t); card.append(d["timing"]["card_ms"])
        except Exception as e:
            errs.append(str(e)[:200])
th = [threading.Thread(target=one) for _ in range(N)]
t0 = time.time()
[x.start() for x in th]; [x.join() for x in th]
w = time.time() - t0
print("YOLO: %d frames in %.1f s = %.2f frames/s, latency mean %.2f s, card-side per frame mean %.0f ms, errors %d %s"
      % (len(done), w, len(done) / w, sum(lat) / max(len(lat), 1), sum(card) / max(len(card), 1), len(errs), errs[:2]))
