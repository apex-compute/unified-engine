"""Smoke test for a running OllApeX: a question, a follow-up (should reuse what the cards hold),
a file, and sampling.

    python ox_test.py [chat] [file] [sample]
"""
import json
import os
import sys
import urllib.parse
import urllib.request

B = "http://127.0.0.1:%s" % os.environ.get("OLLAPEX_PORT", "3000")


def chat(messages, **options):
    req = urllib.request.Request(B + "/api/chat", json.dumps({"messages": messages, "options": options}).encode(),
                                 {"Content-Type": "application/json"})
    text, stats = "", None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            o = json.loads(line)
            if o["type"] == "content":
                text += o["text"]
            elif o["type"] == "error":
                print("ERROR:", o["message"][:600])
            elif o["type"] == "done":
                stats = o["stats"]
    return text, stats


def show(tag, text, s):
    print("=== %s ===" % tag)
    print(text[:700])
    if s:
        print("-> prompt %d tokens (already on cards %d, read %d, %.1f s) | generated %d tokens, %.2f tokens/s, %.0f ms/token | "
              "first token %.1f s | finish %s | context %d/%d | slowest card %.0f ms" % (
                  s["prompt_tokens"], s["reused_tokens"], s["prefill_tokens"], s["prefill_s"], s["eval_tokens"],
                  s["eval_tok_s"], s["ms_per_token"], s["ttft_s"], s["done_reason"], s["context_used"], s["num_ctx"],
                  s.get("slowest_card_compute_ms_per_token", 0)), flush=True)


which = sys.argv[1:] or ["chat", "file", "sample"]
if "chat" in which:
    m = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    t, s = chat(m, num_predict=60)
    show("question", t, s)
    m += [{"role": "assistant", "content": t}, {"role": "user", "content": "And roughly what is its population?"}]
    t, s = chat(m, num_predict=60)
    show("follow-up question", t, s)
if "file" in which:
    body = ("Meeting notes - 10 October\n\nAttendees: Alex, Sam, Jordan.\n1. There are 11 FPGA cards, all working.\n"
            "2. The model generates 2.1 tokens per second.\n3. Next step: demo interface and image support.\n"
            "Decision: the demo will be held on Friday at 14:00.\n").encode()
    req = urllib.request.Request(B + "/api/file", body, {"X-Filename": urllib.parse.quote("meeting notes.txt")})
    f = json.load(urllib.request.urlopen(req, timeout=60))
    print("file:", f["name"], f["kind"], f["chars"], "characters,", f["tokens"], "tokens")
    t, s = chat([{"role": "user", "content": "File: %s\n\n%s\n\n---\n\nWhen will the demo be held and who attended?" % (f["name"], f["text"])}],
                num_predict=80)
    show("file question", t, s)
if "sample" in which:
    for seed in (1, 2):
        t, s = chat([{"role": "user", "content": "Write a two-line poem about the sea."}], num_predict=40, temperature=0.9, top_k=4, seed=seed)
        show("sampling, seed %d" % seed, t, s)
