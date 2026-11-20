"""The portal chat through the API: picture + YOLO context + question, then a follow-up."""
import json, sys, time, urllib.request
B = "http://127.0.0.1:3000"
ctx = open(sys.argv[1]).read().strip(); img = open(sys.argv[2], "rb").read()
f = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/api/file", data=img, headers={"X-Filename": "frame.jpg"})).read())
def chat(msgs):
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(B + "/api/chat", data=json.dumps({"model": "gemma4:26b", "messages": msgs,
        "think": False, "options": {"temperature": 0, "num_predict": 300}}).encode(), headers={"Content-Type": "application/json"}), timeout=900)
    text, st = "", {}
    for line in r:
        ev = json.loads(line)
        if ev.get("type") == "content": text += ev["text"]
        elif ev.get("type") == "done": st = ev["stats"]
        elif ev.get("type") == "error": print("ERROR", ev); sys.exit(1)
    print("ANSWER:", text.strip())
    print("  [prompt %s tok, reused %s, prefill %.1f s, %s tok at %.1f tok/s, wall %.1f s]" % (st.get("prompt_tokens"), st.get("reused_tokens"),
          st.get("prefill_s", 0), st.get("eval_tokens"), st.get("eval_tok_s", 0), time.time() - t0))
    return text
q1 = "How many boats are there, and where are most of them? Is this a marina or a commercial port?"
m = [{"role": "user", "content": ctx + "\n\nQuestion: " + q1, "images": [f["image_id"]]}]
a1 = chat(m)
m += [{"role": "assistant", "content": a1}, {"role": "user", "content": "Which part of the image has the fewest boats?"}]
chat(m)
