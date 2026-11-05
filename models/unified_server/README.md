# Unified model server

One web page for the models that run on the board, picked from a dropdown:

| Model | Input | Runs on the FPGA |
|---|---|---|
| Gemma 3 1B | text | prefill (engine 0) + decode (N-sharded) |
| Gemma 4 E2B | text, image, audio | vision encoder, audio encoder, prefill, decode |
| Gemma 4 E4B | text, image, audio | same code as E2B with E4B's geometry |
| FLUX.2 [klein] 4B | text → image | text encoder, diffusion transformer, VAE decoder |

```bash
# from the repo root
~/flux_env/bin/python models/unified_server/unified_server.py --port 8080
~/flux_env/bin/python models/unified_server/unified_server.py --host 0.0.0.0 --port 8080 --engines 8
```

| Flag | Default | Notes |
|---|---|---|
| `--engines` | 8 | engines for the chat models (FLUX uses each compiled config's own count) |
| `--dev` | xdma0 | DMA device |
| `--python` | this interpreter | interpreter for the runners |
| `--no-reload` | off | FLUX: tick "keep the board loaded" by default |

## How it works

The board is one resource, so a single scheduler thread runs every request in
order, and at most one process holds the device:

- **Chat models** run in a resident `llm_worker.py` process that loads the
  model once (weights to DRAM, program image) and then answers messages. The
  first message to a model pays the load; later messages start prefill at
  once. Picking another model, a FLUX image, or "release the board" stops it.
  The worker speaks JSON lines on a private stdout (`ready`, `token`, `status`,
  `result`, `error`); everything the model code prints goes to stderr, which
  the page shows as the run log. Tokens are pushed to the page as server-sent
  events (`/api/jobs/<id>/stream`, polling as a fallback), revealed frame by
  frame and rendered as Markdown with math (marked, DOMPurify and KaTeX,
  fetched once from the CDN and served from `/vendor/`, so the browser needs
  no internet access; without them answers show as plain text). Stop ends the
  generation after the current token.
- **FLUX.2 klein** runs exactly as in `flux2_klein_4b_server.py` (its module is
  imported): one `flux2_klein_4b_run_from_bin.py` process per image, or its
  resident `--serve` runner with "keep the board loaded".

Prompt handling:

- Text turns carry the conversation (each model's chat template). Gemma 3's
  prefill program takes at most 193 prompt tokens and its context is 512, so
  the oldest turns are dropped to fit (the page says so). Gemma 4 has an
  8192-token context and prefills longer prompts in 512-row chunks.
- An image and/or audio turn is sent on its own: the encoders build a single
  user turn of `[audio] [image] text`. Audio may be uploaded (WAV / FLAC / OGG,
  anything else through `ffmpeg` if installed) or recorded in the browser
  (sent as 16 kHz mono WAV; the browser allows the microphone on localhost or
  https only).
- Gemma 4's multi-engine prefill shards the prompt rows, so a prompt shorter
  than the engine count ("hi") gets a one-line system preamble.
- Gemma 4's prefill program is rebuilt in-process when the prompt length
  changes (a few seconds, weights stay on the board); the metrics report it
  separately and keep it out of the TTFT.

## Metrics

Every answer shows (click an earlier answer to see its own):

- prompt / generated tokens, TTFT on the engines and on the wall clock,
  decode rate (first token, all tokens on the engines, wall clock);
- per phase (vision encoder, audio encoder, prefill, decode): tokens, engine
  time (the on-device latency counter), wall time, tokens/s, model FLOPs,
  GFLOPS and the share of the raw peak of the engines that phase runs on
  (128 FLOP/cycle/engine at the HW_INFO clock).

The static card shows the model geometry, the weight file and the quantized
weights in board DRAM, and the program image sizes. FLUX shows the same tables
as its own server.

## API

| Method | Path | Body / result |
|---|---|---|
| GET | `/api/models` | models, static figures, resident process, FLUX configs |
| POST | `/api/chat` | `{model, messages:[{role, content}], image?:{name, data(base64)}, audio?:{...}, max_new_tokens}` → `{id}` |
| POST | `/api/generate` | FLUX: `{prompt, config, seed, no_reload}` → `{id}` |
| GET | `/api/jobs/<id>` | state, streamed `text`, `metrics` (chat) or `result` (FLUX), log |
| GET | `/api/jobs/<id>/stream` | chat: server-sent `delta` events (new text, phase, token count), then `done` with the job |
| POST | `/api/jobs/<id>/stop` | stop a chat generation |
| GET | `/api/jobs/<id>/image.png` | FLUX image |
| POST | `/api/release` | stop the resident process, free the board |

## Prerequisites

The models' own artifacts, built once by their test scripts:
`models/gemma3/gemma3_bin/weights_gemma3_hf.bin`,
`models/gemma4_e2b/gemma4_e2b_bin/params.bin`,
`models/gemma4_e4b/gemma4_e4b_bin/params.bin` (a chat model without them
generates them on first load, which takes minutes), and the FLUX program sets
under `models/flux2_klein_4b/flux2_klein_4b_bin/programs/`.
