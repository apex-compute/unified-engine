"""Image test for OllApeX through the real server path.

Starts a server on OLLAPEX_TEST_PORT (default 3001) and waits for the model, or with
OLLAPEX_EXTERNAL=1 uses the server already running on that port. Then: text chat, image chat,
follow-up on the image, bigger detail levels, and a plain text chat afterwards.

    python ox_test_image.py [text] [low] [follow] [medium] [high] [text2]
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

PORT = int(os.environ.get("OLLAPEX_TEST_PORT", "3001"))
EXTERNAL = os.environ.get("OLLAPEX_EXTERNAL") == "1"       # test a server that is already running
B = "http://127.0.0.1:%d" % PORT
HERE = os.path.dirname(os.path.abspath(__file__))


def get(path):
    return json.load(urllib.request.urlopen(B + path, timeout=10))


def chat(messages, **options):
    req = urllib.request.Request(B + "/api/chat", json.dumps({"messages": messages, "options": options}).encode(),
                                 {"Content-Type": "application/json"})
    text, stats, t0 = "", None, time.time()
    with urllib.request.urlopen(req, timeout=1500) as r:
        for line in r:
            o = json.loads(line)
            if o["type"] == "content":
                text += o["text"]
            elif o["type"] == "status" and o.get("event") == "vision":
                print("STATUS|vision|%s %dx%d %d tokens" % (o["name"], o["width"], o["height"], o["tokens"]), flush=True)
            elif o["type"] == "error":
                print("ERROR|%s" % o["message"][:600], flush=True)
            elif o["type"] == "done":
                stats = o["stats"]
    return text, stats, time.time() - t0


def show(tag, text, s, wall):
    print("ANSWER|%s|%s" % (tag, text.replace("\n", " ")[:600]), flush=True)
    if s:
        print("STATS|%s|prompt %d (image %d, reused %d, read %d in %.1f s)|generated %d at %.2f tok/s|first token %.1f s|total %.1f s|slowest card %.0f ms" % (
            tag, s["prompt_tokens"], s.get("image_tokens", 0), s["reused_tokens"], s["prefill_tokens"], s["prefill_s"], s["eval_tokens"],
            s["eval_tok_s"], s["ttft_s"], wall, s.get("slowest_card_compute_ms_per_token", 0)), flush=True)
        for im in s.get("images_encoded") or []:
            print("ENCODED|%s|%s -> %s|%d patches, %d tokens|card %s %.1f s|build %.1f s|weights %.1f s" % (
                im["name"], im["source"], im["encoded"], im["patches"], im["tokens"], im["card"], im["card_s"], im["compile_s"],
                im["weights_upload_s"]), flush=True)


def upload(path):
    data = open(path, "rb").read()
    req = urllib.request.Request(B + "/api/file", data, {"X-Filename": urllib.parse.quote(os.path.basename(path))})
    f = json.load(urllib.request.urlopen(req, timeout=60))
    print("UPLOAD|%s|%dx%d|%s" % (f["name"], f["width"], f["height"], {k: (v["width"], v["height"], v["tokens"]) for k, v in f["plans"].items()}), flush=True)
    return f


def make_images(d):
    from PIL import Image, ImageDraw, ImageFont
    os.makedirs(d, exist_ok=True)
    im = Image.new("RGB", (512, 512), (255, 255, 255))
    g = ImageDraw.Draw(im)
    g.ellipse((40, 60, 250, 270), fill=(220, 30, 30))
    g.rectangle((290, 250, 480, 440), fill=(30, 160, 60))
    g.polygon([(300, 40), (470, 40), (385, 190)], fill=(30, 60, 220))
    im.save(os.path.join(d, "shapes.png"))
    im = Image.new("RGB", (768, 512), (245, 245, 245))
    g = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
    except Exception:
        f = None
    for i, line in enumerate(("INVOICE 2026-117", "Apex Compute cards: 11", "Total: 4,250 USD", "Due: 15 November")):
        g.text((40, 50 + i * 100), line, fill=(10, 10, 10), font=f)
    im.save(os.path.join(d, "invoice.png"))
    return os.path.join(d, "shapes.png"), os.path.join(d, "invoice.png")


def main():
    which = sys.argv[1:] or ["text", "low", "follow", "medium", "high", "text2"]
    env = dict(os.environ, OLLAPEX_PORT=str(PORT))
    srv = None if EXTERNAL else subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "ollapex_server.py")], env=env,
                                                 stdout=open(os.path.join(HERE, "server.log"), "w"), stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            time.sleep(5)
            if srv is not None and srv.poll() is not None:
                print("ERROR|server exited %s" % srv.returncode)
                return
            try:
                st = get("/api/status")
            except Exception:
                continue
            if st["state"] == "ready":
                break
            if st.get("error") or time.time() - t0 > 600:
                print("ERROR|load: %s" % (st.get("error") or "timeout")[:800])
                return
        i = st["info"]
        print("LOADED|%.0f s|vision %s on card %s|%s" % (time.time() - t0, st["vision"], i.get("vision_card"), st["image_detail"]), flush=True)
        shapes, invoice = make_images(os.path.join(tempfile.gettempdir(), "ollapex_test_img"))
        if "text" in which:
            show("text", *chat([{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], num_predict=40))
        if "low" in which:
            f = upload(shapes)
            m = [{"role": "user", "content": "Describe this image in two sentences.", "images": [f["image_id"]]}]
            t, s, w = chat(m, num_predict=80, image_detail="low")
            show("low", t, s, w)
            if "follow" in which:
                m += [{"role": "assistant", "content": t}, {"role": "user", "content": "What color is the square?"}]
                show("follow-up", *chat(m, num_predict=40, image_detail="low"))
                show("same image again", *chat([{"role": "user", "content": "How many shapes are in this image? Answer with a number.",
                                                 "images": [f["image_id"]]}], num_predict=20, image_detail="low"))
        if "medium" in which:
            f = upload(invoice)
            show("medium", *chat([{"role": "user", "content": "Read the text in this image.", "images": [f["image_id"]]}],
                                 num_predict=80, image_detail="medium"))
        if "high" in which:
            f = upload(invoice)
            show("high", *chat([{"role": "user", "content": "What is the total amount and when is it due?", "images": [f["image_id"]]}],
                               num_predict=60, image_detail="high"))
        if "text2" in which:
            show("text after images", *chat([{"role": "user", "content": "Name three planets of the solar system."}], num_predict=40))
    finally:
        if srv is None:
            return
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(timeout=60)
        except Exception:
            srv.kill()
        print("SERVER_STOPPED|rc %s" % srv.returncode, flush=True)


if __name__ == "__main__":
    main()
