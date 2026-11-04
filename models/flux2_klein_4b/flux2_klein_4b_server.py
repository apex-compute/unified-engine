#!/usr/bin/env python3
"""Minimal web UI for FLUX.2 [klein] 4B on the board.

Standard library only. Serves one page (prompt box, resolution picker, seed,
Generate) and runs ``flux2_klein_4b_run_from_bin.py`` as a subprocess for each
request, one at a time (the board is a single resource). The picker lists the
program sets found under ``flux2_klein_4b_bin/programs/``; compile more with
``flux2_klein_4b_test.py --compile-only``.

    python models/flux2_klein_4b/flux2_klein_4b_server.py --port 8080
    python models/flux2_klein_4b/flux2_klein_4b_server.py --port 8080 --cpus 0,2-23   # pin the runs

API: GET /api/configs, POST /api/generate {prompt, config, seed}, GET /api/jobs/<id>,
GET /api/jobs/<id>/image.png, GET /api/jobs (history).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUNNER = os.path.join(HERE, "flux2_klein_4b_run_from_bin.py")
PROGRAMS_DIR = os.path.join(HERE, "flux2_klein_4b_bin", "programs")
OUT_DIR = os.path.join(HERE, "flux2_klein_4b_bin", "server")
MAX_PROMPT_CHARS = 2000


def scan_configs() -> list[dict]:
    """One entry per compiled program set: <HxWxTOKENSxSTEPS>_e<N>.json."""
    out = []
    for meta_p in sorted(glob.glob(os.path.join(PROGRAMS_DIR, "*_e*.json"))):
        bin_p = meta_p[:-5] + ".bin"
        if not os.path.exists(bin_p):
            continue
        try:
            with open(meta_p) as f:
                meta = json.load(f)
            g = meta["geometry"]
            out.append({
                "id": os.path.basename(meta_p)[:-5],
                "height": int(g["height"]), "width": int(g["width"]),
                "max_seq_len": int(g["max_sequence_length"]), "steps": int(g["steps"]),
                "engines": int(meta["engines"]),
            })
        except (OSError, KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda c: (c["height"] * c["width"], c["max_seq_len"], c["engines"]))
    return out


class Job:
    def __init__(self, prompt: str, config: dict, seed: int):
        self.id = uuid.uuid4().hex[:12]
        self.prompt = prompt
        self.config = config
        self.seed = seed
        self.state = "queued"       # queued | running | done | error
        self.log: list[str] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.image = os.path.join(OUT_DIR, f"{self.id}.png")

    def public(self) -> dict:
        now = time.time()
        return {
            "id": self.id, "prompt": self.prompt, "config": self.config, "seed": self.seed,
            "state": self.state, "error": self.error, "result": self.result,
            "log": "".join(self.log[-200:]),
            "elapsed": round((self.finished or now) - (self.started or now), 1) if self.started else 0.0,
            "image": f"/api/jobs/{self.id}/image.png" if self.state == "done" else None,
        }


class Runner(threading.Thread):
    """Runs queued jobs sequentially through the run-from-bin script."""

    def __init__(self, python: str, cpus: str | None, dev: str):
        super().__init__(daemon=True)
        self.python, self.cpus, self.dev = python, cpus, dev
        self.q: "queue.Queue[Job]" = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()

    def submit(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put(job)

    def position(self, job: Job) -> int:
        with self.lock:
            ahead = [j for j in self.jobs.values() if j.state == "queued" and j.created < job.created]
        return len(ahead)

    def command(self, job: Job) -> list[str]:
        c = job.config
        cmd = []
        if self.cpus:
            cmd += ["taskset", "-c", self.cpus]
        cmd += [self.python, RUNNER,
                "--height", str(c["height"]), "--width", str(c["width"]),
                "--max-seq-len", str(c["max_seq_len"]), "--steps", str(c["steps"]),
                "--engines", str(c["engines"]), "--seed", str(job.seed), "--dev", self.dev,
                f"--prompt={job.prompt}", "--out", job.image]
        return cmd

    def run(self) -> None:
        while True:
            job = self.q.get()
            job.state, job.started = "running", time.time()
            try:
                os.makedirs(OUT_DIR, exist_ok=True)
                proc = subprocess.Popen(self.command(job), cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    job.log.append(line)
                    if line.startswith("TEST_RESULT: "):
                        try:
                            job.result = json.loads(line[len("TEST_RESULT: "):])
                        except json.JSONDecodeError:
                            pass
                rc = proc.wait()
                if rc != 0:
                    last = next((l.strip() for l in reversed(job.log) if l.strip()), "")
                    raise RuntimeError(f"runner exited with code {rc}: {last}")
                if not os.path.exists(job.image):
                    raise RuntimeError("runner finished but wrote no image")
                job.state = "done"
            except Exception as e:  # noqa: BLE001 - report anything to the page
                job.state, job.error = "error", str(e)
            finally:
                job.finished = time.time()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FLUX.2 klein on the Unified Engine</title>
<style>
  :root { --bg:#f6f7f9; --fg:#1c1e21; --muted:#6b7280; --card:#fff; --line:#d9dde3; --accent:#2563eb; --err:#b91c1c; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111317; --fg:#e6e8eb; --muted:#9aa3ad; --card:#1a1d23; --line:#2d323b; --accent:#5b8cff; --err:#f87171; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
  main { max-width: 960px; margin: 0 auto; padding: 24px 16px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px; }
  label { display:block; font-size:13px; color:var(--muted); margin-bottom:4px; }
  textarea, select, input { width:100%; font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px; }
  textarea { min-height: 72px; resize: vertical; }
  .row { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; align-items:flex-end; }
  .row > div { flex: 1 1 180px; }
  button { font:inherit; padding:9px 18px; border:0; border-radius:6px; background:var(--accent); color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .status { margin-top:12px; color:var(--muted); min-height: 1.4em; }
  .status.err { color: var(--err); }
  pre { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px; max-height:220px; overflow:auto; font-size:12px; white-space:pre-wrap; }
  #result img { max-width:100%; height:auto; border-radius:8px; border:1px solid var(--line); display:block; }
  .meta { color:var(--muted); font-size:13px; margin-top:8px; }
  .hist { display:flex; gap:8px; flex-wrap:wrap; }
  .hist img { width:96px; height:96px; object-fit:cover; border-radius:6px; border:1px solid var(--line); cursor:pointer; }
  details summary { cursor:pointer; color:var(--muted); font-size:13px; }
</style>
</head>
<body>
<main>
  <h1>FLUX.2 [klein] 4B</h1>
  <p class="sub">Text-to-image on the Unified Engine (Alveo U50). Every phase runs on the FPGA; the host only tokenizes and writes the PNG.</p>

  <div class="card">
    <label for="prompt">Prompt</label>
    <textarea id="prompt">A photo of a red sports car on a mountain road at sunset</textarea>
    <div class="row">
      <div><label for="config">Resolution</label><select id="config"></select></div>
      <div style="flex:0 1 120px"><label for="seed">Seed</label><input id="seed" type="number" value="0" min="0"></div>
      <div style="flex:0 0 auto"><button id="go">Generate</button></div>
    </div>
    <div id="status" class="status"></div>
    <details><summary>Run log</summary><pre id="log"></pre></details>
  </div>

  <div class="card" id="result" hidden>
    <img id="img" alt="generated image">
    <div class="meta" id="meta"></div>
  </div>

  <div class="card" id="history" hidden>
    <label>Earlier images (this server session)</label>
    <div class="hist" id="hist"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let configs = [], current = null, timer = null;

function fmt(c) {
  return `${c.width}×${c.height} — ${c.max_seq_len} text tokens, ${c.steps} steps, ${c.engines} engines`;
}
async function loadConfigs() {
  configs = await (await fetch('/api/configs')).json();
  const sel = $('config'); sel.innerHTML = '';
  for (const c of configs) { const o = document.createElement('option'); o.value = c.id; o.textContent = fmt(c); sel.appendChild(o); }
  if (!configs.length) { $('status').textContent = 'No compiled program sets found under flux2_klein_4b_bin/programs/.'; $('go').disabled = true; }
  else { sel.value = (configs.find(c => c.width === 256 && c.max_seq_len === 64) || configs[0]).id; }
}
function showJob(j) {
  if (!j || j.state !== 'done') return;
  $('img').src = j.image + '?t=' + (j.result && j.result.total_s || 0);
  const r = j.result || {};
  $('meta').textContent = `"${j.prompt}" — ${j.config.width}×${j.config.height}, seed ${j.seed}` +
    (r.total_s ? ` — ${r.total_s.toFixed(1)} s on the board (TE ${r.te_s.toFixed(1)} s, DiT ${r.dit_s.toFixed(1)} s, VAE ${r.vae_s.toFixed(1)} s)` : '');
  $('result').hidden = false;
}
async function loadHistory() {
  const jobs = (await (await fetch('/api/jobs')).json()).filter(j => j.state === 'done');
  if (jobs.length && current === null) showJob(jobs[jobs.length - 1]);
  $('history').hidden = jobs.length < 2;
  $('hist').innerHTML = '';
  for (const j of jobs.slice().reverse()) {
    const im = document.createElement('img'); im.src = j.image; im.title = j.prompt;
    im.onclick = () => showJob(j); $('hist').appendChild(im);
  }
}
async function poll() {
  const j = await (await fetch('/api/jobs/' + current)).json();
  $('log').textContent = j.log;
  if (j.state === 'queued') $('status').textContent = `Queued (${j.position} ahead)…`;
  else if (j.state === 'running') $('status').textContent = `Running on the board… ${j.elapsed.toFixed(0)} s`;
  else {
    clearInterval(timer); timer = null; $('go').disabled = false;
    $('status').className = 'status' + (j.state === 'error' ? ' err' : '');
    $('status').textContent = j.state === 'error' ? 'Failed: ' + j.error : `Done in ${j.elapsed.toFixed(1)} s.`;
    showJob(j); loadHistory();
  }
}
$('go').onclick = async () => {
  const body = { prompt: $('prompt').value, config: $('config').value, seed: parseInt($('seed').value || '0', 10) };
  $('go').disabled = true; $('status').className = 'status'; $('status').textContent = 'Submitting…'; $('log').textContent = '';
  const r = await fetch('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { $('status').className = 'status err'; $('status').textContent = j.error || 'request rejected'; $('go').disabled = false; return; }
  current = j.id; poll(); timer = setInterval(poll, 1000);
};
loadConfigs(); loadHistory();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    runner: Runner  # set by serve()

    def log_message(self, fmt, *args):  # quieter than the default
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/configs":
            return self._json(scan_configs())
        if path == "/api/jobs":
            with self.runner.lock:
                jobs = [self.runner.jobs[i] for i in self.runner.order]
            return self._json([{k: v for k, v in j.public().items() if k != "log"} for j in jobs])
        if path.startswith("/api/jobs/"):
            parts = path[len("/api/jobs/"):].split("/")
            job = self.runner.jobs.get(parts[0])
            if job is None:
                return self._json({"error": "no such job"}, HTTPStatus.NOT_FOUND)
            if len(parts) == 1:
                d = job.public()
                d["position"] = self.runner.position(job)
                return self._json(d)
            if parts[1] == "image.png" and job.state == "done":
                with open(job.image, "rb") as f:
                    return self._send(200, f.read(), "image/png")
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            prompt = str(body.get("prompt", "")).strip()
            seed = int(body.get("seed", 0))
            cfg = next((c for c in scan_configs() if c["id"] == body.get("config")), None)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
        if not prompt:
            return self._json({"error": "empty prompt"}, HTTPStatus.BAD_REQUEST)
        if len(prompt) > MAX_PROMPT_CHARS:
            return self._json({"error": f"prompt longer than {MAX_PROMPT_CHARS} characters"}, HTTPStatus.BAD_REQUEST)
        if cfg is None:
            return self._json({"error": "unknown configuration"}, HTTPStatus.BAD_REQUEST)
        if seed < 0:
            return self._json({"error": "seed must be >= 0"}, HTTPStatus.BAD_REQUEST)
        job = Job(prompt, cfg, seed)
        self.runner.submit(job)
        self._json({"id": job.id, "position": self.runner.position(job)}, HTTPStatus.ACCEPTED)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Minimal web UI for FLUX.2 klein 4B on the board")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to reach it from other machines)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the run script")
    ap.add_argument("--cpus", default=None, help="pin each run with taskset -c <list>, e.g. 0,2-23")
    ap.add_argument("--dev", default="xdma0")
    args = ap.parse_args(argv)

    cfgs = scan_configs()
    if not cfgs:
        print(f"WARNING: no program sets under {PROGRAMS_DIR}; compile some with flux2_klein_4b_test.py --compile-only",
              file=sys.stderr)
    for c in cfgs:
        print(f"  {c['id']}: {c['width']}x{c['height']}, {c['max_seq_len']} text tokens, {c['steps']} steps, "
              f"{c['engines']} engines", file=sys.stderr)
    runner = Runner(args.python, args.cpus, args.dev)
    runner.start()
    Handler.runner = runner
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FLUX.2 klein server on http://{args.host}:{args.port}/  (runner: {args.python}"
          f"{', taskset -c ' + args.cpus if args.cpus else ''})", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
