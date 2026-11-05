#!/usr/bin/env python3
"""Minimal web UI for FLUX.2 [klein] 4B on the board.

Standard library only. Serves one page (prompt box, resolution picker, seed,
Generate) and runs ``flux2_klein_4b_run_from_bin.py`` for each request, one at
a time (the board is a single resource). The picker lists the program sets
found under ``flux2_klein_4b_bin/programs/``; compile more with
``flux2_klein_4b_test.py --compile-only``. Selecting one shows its static
figures (weight images, program sizes, FLOPs per phase); a finished run shows
every phase's on-device FLOP rate against the raw peak of the engines.

"Keep the board loaded" (the page's checkbox; ``--no-reload`` makes it the
default) runs the jobs through one long-lived ``--serve`` runner that keeps the
device, the tokenizer, the VAE weights and the program set of the last
configuration, so a run skips process start-up, the host file reads and the
program DMA; the token embedding table is sent to the board once per runner and
stays resident. The TE / DiT weight images are still sent on every run (they
and the VAE arena time-share the params arena below the 4 GiB DMA window).
While that runner lives it holds the device, so other board users wait;
"release the board" on the page (or an unchecked run) stops it.

    python models/flux2_klein_4b/flux2_klein_4b_server.py --port 8080
    python models/flux2_klein_4b/flux2_klein_4b_server.py --port 8080 --cpus 0,2-23 --no-reload

API: GET /api/configs, POST /api/generate {prompt, config, seed, no_reload}, GET /api/jobs/<id>,
GET /api/jobs/<id>/image.png, GET /api/jobs (history), POST /api/release (stop the resident runner).
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
CONFIG = os.path.join(HERE, "flux2_klein_4b_config.json")
BIN_DIR = os.path.join(HERE, "flux2_klein_4b_bin")
PROGRAMS_DIR = os.path.join(BIN_DIR, "programs")
OUT_DIR = os.path.join(BIN_DIR, "server")
BOARD_CACHE = os.path.join(OUT_DIR, "board.json")   # clock / peak seen on the last run
MAX_PROMPT_CHARS = 2000
PHASES = ("te", "dit", "vae")


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def weight_files() -> dict:
    """Sizes of the geometry-independent weight images (flux2_klein_4b_weights.py)."""
    with open(CONFIG) as f:
        cfg = json.load(f)
    paths = cfg["paths"]
    return {
        "te": _size(os.path.join(HERE, paths["params_te_bin"])),
        "dit": _size(os.path.join(HERE, paths["params_dit_bin"])),
        "vae": _size(os.path.join(HERE, paths["vae_weights"])),
        "embed_tokens": _size(os.path.join(HERE, paths["embed_tokens_bin"])),
        "vae_precision": cfg.get("quantization", {}).get("vae_precision", "if8"),
    }


def scan_configs() -> list[dict]:
    """One entry per compiled program set: <HxWxTOKENSxSTEPS>_e<N>.json, with its static figures."""
    out = []
    for meta_p in sorted(glob.glob(os.path.join(PROGRAMS_DIR, "*_e*.json"))):
        bin_p = meta_p[:-5] + ".bin"
        if not os.path.exists(bin_p):
            continue
        try:
            with open(meta_p) as f:
                meta = json.load(f)
            g = meta["geometry"]
            images = meta.get("images", [])
            phases = {}
            for n in PHASES:
                if f"{n}_flops" not in meta:
                    continue
                primary = int(meta[f"{n}_program_size"])             # engine 0's segment of this phase
                workers = sum(int(im["size"]) for im in images if im.get("phase") == n)
                phases[n] = {"program_bytes": primary, "worker_bytes": workers,
                             "all_engines_bytes": primary + workers, "flops": int(meta[f"{n}_flops"])}
            # anything else in the bin (the per-engine barrier flag-clear stubs), engine 0 / the others
            other = [im for im in images if im.get("phase") not in PHASES + ("primary",)]
            other_primary = sum(int(im["size"]) for im in other if int(im.get("engine", 0)) == 0)
            other_workers = sum(int(im["size"]) for im in other) - other_primary
            out.append({
                "id": os.path.basename(meta_p)[:-5],
                "height": int(g["height"]), "width": int(g["width"]),
                "max_seq_len": int(g["max_sequence_length"]), "steps": int(g["steps"]),
                "engines": int(meta["engines"]),
                "image_tokens": (int(g["height"]) // 16) * (int(g["width"]) // 16),
                "program_bin_bytes": _size(bin_p),
                "program_images_bytes": sum(int(im["size"]) for im in images),
                "phases": phases,
                "other_images": {"program_bytes": other_primary, "worker_bytes": other_workers,
                                 "all_engines_bytes": other_primary + other_workers, "count": len(other)},
                "total_flops": sum(p["flops"] for p in phases.values()),
                "params_image_bytes": int(meta.get("params_image_size", 0)),
                "tensor_dram_bytes": int(meta.get("tensor_dram_usage", 0)),
                "vae_const_bytes": int(meta.get("vae", {}).get("const_size", 0)),
                "compile_seconds": meta.get("compile_seconds"),
            })
        except (OSError, KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda c: (c["height"] * c["width"], c["max_seq_len"], c["engines"]))
    return out


def load_board() -> dict:
    try:
        with open(BOARD_CACHE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_board(result: dict) -> None:
    if not result.get("peak_gflops"):
        return
    board = {"clock_mhz": result["clock_mhz"], "peak_gflops_per_engine": result["peak_gflops_per_engine"],
             "seen": time.time()}
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(BOARD_CACHE, "w") as f:
            json.dump(board, f)
    except OSError:
        pass


class Job:
    def __init__(self, prompt: str, config: dict, seed: int, no_reload: bool):
        self.id = uuid.uuid4().hex[:12]
        self.prompt = prompt
        self.config = config
        self.seed = seed
        self.no_reload = no_reload
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
            "no_reload": self.no_reload, "state": self.state, "error": self.error, "result": self.result,
            "log": "".join(self.log[-200:]),
            "elapsed": round((self.finished or now) - (self.started or now), 1) if self.started else 0.0,
            "image": f"/api/jobs/{self.id}/image.png" if self.state == "done" else None,
        }


class Worker:
    """A ``--serve`` runner process: holds the device and the host state between jobs."""

    def __init__(self, cmd: list[str], engines: int):
        self.engines = engines
        self.loaded: str | None = None
        self.started = time.time()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()               # EOF ends the job loop
            self.proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()


class Runner(threading.Thread):
    """Runs queued jobs sequentially: a one-shot runner process per job, or the resident worker."""

    def __init__(self, python: str, cpus: str | None, dev: str):
        super().__init__(daemon=True)
        self.python, self.cpus, self.dev = python, cpus, dev
        self.q: "queue.Queue[Job]" = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.worker: Worker | None = None
        self.release_requested = False

    def submit(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put(job)

    def position(self, job: Job) -> int:
        with self.lock:
            ahead = [j for j in self.jobs.values() if j.state == "queued" and j.created < job.created]
        return len(ahead)

    def worker_state(self) -> dict | None:
        w = self.worker
        if w is None or not w.alive():
            return None
        return {"engines": w.engines, "loaded": w.loaded, "since": w.started}

    def _prefix(self) -> list[str]:
        return ["taskset", "-c", self.cpus] if self.cpus else []

    def command(self, job: Job) -> list[str]:
        c = job.config
        return self._prefix() + [self.python, RUNNER,
                                 "--height", str(c["height"]), "--width", str(c["width"]),
                                 "--max-seq-len", str(c["max_seq_len"]), "--steps", str(c["steps"]),
                                 "--engines", str(c["engines"]), "--seed", str(job.seed), "--dev", self.dev,
                                 f"--prompt={job.prompt}", "--out", job.image]

    def stop_worker(self) -> None:
        w, self.worker = self.worker, None
        if w is not None:
            w.stop()

    def _ensure_worker(self, job: Job) -> Worker:
        w = self.worker
        if w is not None and (not w.alive() or w.engines != job.config["engines"]):
            self.stop_worker()
            w = None
        if w is None:
            cmd = self._prefix() + [self.python, RUNNER, "--serve", "--engines", str(job.config["engines"]),
                                    "--dev", self.dev]
            w = Worker(cmd, job.config["engines"])
            job.log.append("starting the resident runner (device, tokenizer, token embedding, VAE weights)...\n")
            assert w.proc.stdout is not None
            for line in w.proc.stdout:              # start-up output until the runner is ready
                job.log.append(line)
                if line.startswith("SERVE_READY"):
                    break
            if not w.alive():
                raise RuntimeError("the resident runner exited during start-up")
            self.worker = w
        return w

    def _run_resident(self, job: Job) -> None:
        w = self._ensure_worker(job)
        c = job.config
        spec = {"prompt": job.prompt, "seed": job.seed, "height": c["height"], "width": c["width"],
                "max_seq_len": c["max_seq_len"], "steps": c["steps"], "out": job.image, "reload": False}
        assert w.proc.stdin is not None and w.proc.stdout is not None
        w.proc.stdin.write(json.dumps(spec) + "\n")
        w.proc.stdin.flush()
        for line in w.proc.stdout:
            job.log.append(line)
            if line.startswith("TEST_RESULT: "):
                job.result = json.loads(line[len("TEST_RESULT: "):])
                w.loaded = job.result.get("config")
                return
            if line.startswith("JOB_ERROR: "):
                raise RuntimeError(json.loads(line[len("JOB_ERROR: "):]).get("error", "job failed"))
        self.stop_worker()
        raise RuntimeError("the resident runner exited during the job")

    def _run_oneshot(self, job: Job) -> None:
        self.stop_worker()                          # the resident runner holds the device
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

    def run(self) -> None:
        while True:
            try:
                job = self.q.get(timeout=1.0)
            except queue.Empty:
                if self.release_requested:
                    self.release_requested = False
                    self.stop_worker()
                continue
            job.state, job.started = "running", time.time()
            try:
                os.makedirs(OUT_DIR, exist_ok=True)
                if job.no_reload:
                    self._run_resident(job)
                else:
                    self._run_oneshot(job)
                if not os.path.exists(job.image):
                    raise RuntimeError("runner finished but wrote no image")
                if job.result:
                    save_board(job.result)
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
  :root { --bg:#f6f7f9; --fg:#1c1e21; --muted:#6b7280; --card:#fff; --line:#d9dde3; --accent:#2563eb; --err:#b91c1c; --bar:#dbe4ff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111317; --fg:#e6e8eb; --muted:#9aa3ad; --card:#1a1d23; --line:#2d323b; --accent:#5b8cff; --err:#f87171; --bar:#26324f; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }
  main { max-width: 1240px; margin: 0 auto; padding: 24px 20px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 13px; color: var(--muted); font-weight: 600; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .03em; }
  .sub { color: var(--muted); margin: 0 0 20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:16px; }
  label { display:block; font-size:13px; color:var(--muted); margin-bottom:4px; }
  textarea, select, input[type=number] { width:100%; font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:8px; }
  textarea { min-height: 72px; resize: vertical; }
  .row { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; align-items:flex-end; }
  .row > div { flex: 1 1 220px; }
  .check { display:flex; align-items:center; gap:8px; margin-top:12px; font-size:14px; color:var(--fg); }
  .check input { width:auto; margin:0; }
  .check a { color: var(--accent); font-size: 13px; margin-left: 8px; }
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
  .block + .block { margin-top: 18px; }
  .narrow { max-width: 620px; }
  .tw { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; white-space: normal; min-width: 160px; }
  th { color: var(--muted); font-weight: 500; }
  tr.total td { font-weight: 600; border-bottom: 0; }
  td.util { position: relative; min-width: 90px; }
  td.util span { position: relative; z-index: 1; }
  td.util i { position:absolute; left:0; top:3px; bottom:3px; background:var(--bar); border-radius:3px; z-index:0; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; line-height: 1.5; }
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
    <div class="check">
      <input type="checkbox" id="noreload" __NO_RELOAD__>
      <label for="noreload" style="margin:0;color:inherit;font-size:14px">Keep the board loaded between runs (no reload)</label>
      <a href="#" id="release" hidden>release the board</a>
    </div>
    <div id="status" class="status"></div>
    <details><summary>Run log</summary><pre id="log"></pre></details>
  </div>

  <div class="card" id="static" hidden>
    <div class="block narrow"><h2>Weights on the board</h2><div class="tw"><table id="weights"></table></div></div>
    <div class="block"><h2>Programs of this configuration</h2><div class="tw"><table id="programs"></table></div></div>
    <div class="note" id="staticnote"></div>
  </div>

  <div class="card" id="result" hidden>
    <img id="img" alt="generated image">
    <div class="meta" id="meta"></div>
    <h2 style="margin-top:16px">Run on the board</h2>
    <div class="tw"><table id="stats"></table></div>
    <div class="note" id="statsnote"></div>
  </div>

  <div class="card" id="history" hidden>
    <label>Earlier images (this server session)</label>
    <div class="hist" id="hist"></div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let configs = [], board = {}, current = null, timer = null;
const NAMES = { te: 'Text encoder (Qwen3-4B)', dit: 'Diffusion transformer', vae: 'VAE decoder' };

const gb = b => b >= 1e9 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(1) + ' MB';
const tf = f => (f / 1e12).toFixed(2) + ' TFLOP';
const pct = x => (100 * x).toFixed(0) + '%';
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function fmt(c) {
  return `${c.width}×${c.height} — ${c.max_seq_len} text tokens, ${c.steps} steps, ${c.engines} engines`;
}
function rows(head, body) {
  return '<tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr>' +
    body.map(r => `<tr${r.total ? ' class="total"' : ''}>` + r.cells.map(c => typeof c === 'object'
      ? `<td class="util"><i style="width:${Math.min(100, 100 * c.util).toFixed(1)}%"></i><span>${pct(c.util)}</span></td>`
      : `<td>${c}</td>`).join('') + '</tr>').join('');
}
function peakOf(c) { return board.peak_gflops_per_engine ? board.peak_gflops_per_engine * c.engines : 0; }

function showStatic() {
  const c = configs.find(x => x.id === $('config').value);
  if (!c) { $('static').hidden = true; return; }
  const w = board.weights || {};
  const wtotal = (w.te || 0) + (w.dit || 0) + (w.vae || 0) + (w.embed_tokens || 0);
  $('weights').innerHTML = rows(['Image', 'Format', 'Where', 'Size'], [
    { cells: ['Text encoder', 'IF4', 'params arena, sent per run', gb(w.te || 0)] },
    { cells: ['Diffusion transformer', 'IF4 + BF16 heads', 'params arena, sent per run', gb(w.dit || 0)] },
    { cells: ['VAE decoder', (w.vae_precision || 'if8').toUpperCase() + (w.vae_precision === 'bf16' ? ' (exact checkpoint weights)' : ''), 'params arena, sent per run', gb(w.vae || 0)] },
    { cells: ['Token embedding table', 'BF16', 'tensor region, resident', gb(w.embed_tokens || 0)] },
    { cells: ['Total on the board', '', '', gb(wtotal)], total: true },
  ]);
  const peak = peakOf(c);
  const body = [];
  for (const n of ['te', 'dit', 'vae']) {
    const p = c.phases[n]; if (!p) continue;
    body.push({ cells: [NAMES[n], gb(p.program_bytes), gb(p.worker_bytes), gb(p.all_engines_bytes), tf(p.flops), peak ? (p.flops / peak / 1e9).toFixed(1) + ' s' : '—'] });
  }
  const o = c.other_images || { program_bytes: 0, worker_bytes: 0, all_engines_bytes: 0, count: 0 };
  if (o.count) body.push({ cells: [`Barrier flag-clear stubs (${o.count} \u00d7 ${o.all_engines_bytes / o.count} B)`, gb(o.program_bytes), gb(o.worker_bytes), gb(o.all_engines_bytes), '', ''] });
  const sum = k => Object.values(c.phases).reduce((a, p) => a + p[k], 0) + o[k];
  body.push({ cells: ['Total (program bin)', gb(sum('program_bytes')), gb(sum('worker_bytes')), gb(c.program_bin_bytes), tf(c.total_flops), peak ? (c.total_flops / peak / 1e9).toFixed(1) + ' s' : '—'], total: true });
  $('programs').innerHTML = rows(['Phase', 'Engine 0', `Engines 1–${c.engines - 1}`, 'All engines', 'Work', 'At peak'], body);
  $('staticnote').textContent =
    `${c.image_tokens} image + ${c.max_seq_len} text tokens. Engine 0 runs the primary program of each phase and every other engine its own worker program; the bin is all of them. ` +
    `a ${gb(c.vae_const_bytes)} VAE constant image is built from the weights per run; tensor DRAM ${gb(c.tensor_dram_bytes)}. ` +
    (peak ? `Raw peak ${peak.toFixed(0)} GFLOPS = ${c.engines} engines × 128 FLOP/cycle at ${board.clock_mhz.toFixed(0)} MHz.`
          : 'The raw peak of the board is known after the first run.');
  $('static').hidden = false;
}
function showWorker() {
  const w = board.worker;
  $('release').hidden = !w;
  if (w) $('release').title = `resident runner on ${w.engines} engines` + (w.loaded ? `, ${w.loaded} loaded` : '');
}
async function loadConfigs() {
  const d = await (await fetch('/api/configs')).json();
  configs = d.configs; board = d.board;
  const sel = $('config'), keep = sel.value; sel.innerHTML = '';
  for (const c of configs) { const o = document.createElement('option'); o.value = c.id; o.textContent = fmt(c); sel.appendChild(o); }
  if (!configs.length) { $('status').textContent = 'No compiled program sets found under flux2_klein_4b_bin/programs/.'; $('go').disabled = true; }
  else { sel.value = configs.some(c => c.id === keep) ? keep : (configs.find(c => c.width === 256 && c.max_seq_len === 64) || configs[0]).id; }
  showStatic(); showWorker();
}
function showJob(j) {
  if (!j || j.state !== 'done') return;
  $('img').src = j.image + '?t=' + (j.result && j.result.total_s || 0);
  const r = j.result || {};
  $('meta').innerHTML = `“${esc(j.prompt)}” — ${j.config.width}×${j.config.height}, seed ${j.seed}` +
    (r.device_s ? ` — ${r.device_s.toFixed(1)} s of engine time` : (r.total_s ? ` — ${r.total_s.toFixed(1)} s` : '')) +
    (r.reload ? ` — setup ${r.setup_s.toFixed(1)} s: programs ${r.reload.programs ? 'reloaded' : 'kept on the board'}, ` +
                `TE + DiT weight images (${gb(r.reload.weight_bytes)}) re-sent` +
                (r.reload.embed === undefined ? '' : `, token embedding table ${r.reload.embed ? 'sent' : 'already resident'}`) : '');
  const body = [];
  if (r.phases) {
    for (const n of ['te', 'dit', 'vae']) {
      const p = r.phases[n]; if (!p) continue;
      body.push({ cells: [NAMES[n], tf(p.flops), p.device_s.toFixed(2) + ' s', p.gflops.toFixed(0), { util: p.utilization }] });
    }
    body.push({ cells: ['Total', tf(r.total_flops), r.device_s.toFixed(2) + ' s', r.total_gflops.toFixed(0), { util: r.utilization }], total: true });
    $('stats').innerHTML = rows(['Phase', 'Work', 'Engine time', 'GFLOPS', 'Of raw peak'], body);
    $('statsnote').textContent =
      `Raw peak ${r.peak_gflops.toFixed(0)} GFLOPS (${r.engines} engines × 128 FLOP/cycle at ${r.clock_mhz.toFixed(0)} MHz). ` +
      `Every rate uses the engines' own execution time (the on-device latency counter of each phase); the total is the sum of the three. ` +
      `The host side (weight DMA, tokenizing, the PNG) is outside these numbers: ${(r.wall_s || r.total_s).toFixed(1)} s wall for the whole run. ` +
      `The TE and DiT weight images cannot stay resident: they and the VAE arena time-share the params arena below the 4 GiB DMA window.`;
  } else {
    $('stats').innerHTML = ''; $('statsnote').textContent = r.total_s ? `TE ${r.te_s.toFixed(1)} s, DiT ${r.dit_s.toFixed(1)} s, VAE ${r.vae_s.toFixed(1)} s.` : '';
  }
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
    loadConfigs();                              // the board's clock / peak and the runner state
  }
}
$('config').onchange = showStatic;
$('release').onclick = async e => {
  e.preventDefault();
  await fetch('/api/release', { method: 'POST' });
  $('status').className = 'status'; $('status').textContent = 'Resident runner stopped; the board is free.';
  setTimeout(loadConfigs, 1500);
};
$('go').onclick = async () => {
  const body = { prompt: $('prompt').value, config: $('config').value, seed: parseInt($('seed').value || '0', 10),
                 no_reload: $('noreload').checked };
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
    runner: Runner          # set by main()
    page: bytes = b""

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
            return self._send(200, self.page, "text/html; charset=utf-8")
        if path == "/api/configs":
            board = load_board()
            board["weights"] = weight_files()
            board["worker"] = self.runner.worker_state()
            return self._json({"configs": scan_configs(), "board": board})
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
        if self.path == "/api/release":
            self.runner.release_requested = True       # the runner thread stops it between jobs
            return self._json({"ok": True}, HTTPStatus.ACCEPTED)
        if self.path != "/api/generate":
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            prompt = str(body.get("prompt", "")).strip()
            seed = int(body.get("seed", 0))
            no_reload = bool(body.get("no_reload", False))
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
        job = Job(prompt, cfg, seed, no_reload)
        self.runner.submit(job)
        self._json({"id": job.id, "position": self.runner.position(job)}, HTTPStatus.ACCEPTED)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Minimal web UI for FLUX.2 klein 4B on the board")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to reach it from other machines)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the run script")
    ap.add_argument("--cpus", default=None, help="pin each run with taskset -c <list>, e.g. 0,2-23")
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--no-reload", action="store_true",
                    help="tick 'keep the board loaded between runs' by default (a resident runner holds the device)")
    args = ap.parse_args(argv)

    cfgs = scan_configs()
    if not cfgs:
        print(f"WARNING: no program sets under {PROGRAMS_DIR}; compile some with flux2_klein_4b_test.py --compile-only",
              file=sys.stderr)
    for c in cfgs:
        print(f"  {c['id']}: {c['width']}x{c['height']}, {c['max_seq_len']} text tokens, {c['steps']} steps, "
              f"{c['engines']} engines, {c['total_flops'] / 1e12:.2f} TFLOP, bin {c['program_bin_bytes'] / 1e6:.1f} MB",
              file=sys.stderr)
    runner = Runner(args.python, args.cpus, args.dev)
    runner.start()
    Handler.runner = runner
    Handler.page = PAGE.replace("__NO_RELOAD__", "checked" if args.no_reload else "").encode()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"FLUX.2 klein server on http://{args.host}:{args.port}/  (runner: {args.python}"
          f"{', taskset -c ' + args.cpus if args.cpus else ''}{', no-reload by default' if args.no_reload else ''})",
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop_worker()                        # free the device
    return 0


if __name__ == "__main__":
    sys.exit(main())
