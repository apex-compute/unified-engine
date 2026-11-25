# Unified model server

One web page for the models that run on the board, picked from a dropdown:

| Model | Input | Runs on the FPGA |
|---|---|---|
| Gemma 3 1B | text | prefill + decode (both N-sharded), 512-token context |
| Gemma 4 E2B | text, image, audio | vision encoder, audio encoder, prefill, decode |
| Gemma 4 E4B | text, image, audio | same code as E2B with E4B's geometry |
| Qwen2.5-Omni 7B | text, image, audio | vision encoder, audio encoder, Thinker prefill + decode (all eight engines) |
| Qwen3.5 9B | text, image | five cards, 16,384-token context |
| Qwen3.8 27B | text, image | five cards, 8,192-token context |
| GPT-OSS 20B | text | five cards, 8,192-token context, streams the Harmony final answer |
| Gemma 4 26B-A4B | text, image | five cards, 8,192-token context, tensor-parallel vision encoder |
| YOLO26s-OBB | image | oriented object detection on one engine, returns an annotated image |
| FLUX.2 [klein] 4B | text → image | text encoder, diffusion transformer, VAE decoder |

```bash
# from the repo root
~/flux_env/bin/python models/unified_server/unified_server.py --port 8080
~/flux_env/bin/python models/unified_server/unified_server.py --host 0.0.0.0 --port 8080 --engines 8
```

| Flag | Default | Notes |
|---|---|---|
| `--engines` | 8 | engines for the single-card chat models |
| `--dev` | xdma0 | device for single-card models and YOLO26 |
| `--tp-devices` | first five cards in PCI order | five comma-separated XDMA names in shard order, also configurable through `APEX_TP_DEVICES` |
| `--python` | this interpreter | interpreter for the runners |
| `--reload-per-image` | off | FLUX: untick "keep the board loaded" by default (weights reloaded per image) |

## How it works

The board is one resource, so a single scheduler thread runs every request in
order. A five-card job owns the cards as a group.

- **Chat models** run in a resident `llm_worker.py` process that loads the
  model once (weights to DRAM, program image) and then answers messages. The
  first message to a model pays the load; later messages start prefill at
  once. Running another model or using "release the board" stops it.
  The worker speaks JSON lines on a private stdout (`ready`, `token`, `status`,
  `result`, `error`); everything the model code prints goes to stderr, which
  the page shows as the run log. Tokens are pushed to the page as server-sent
  events (`/api/jobs/<id>/stream`, polling as a fallback), revealed frame by
  frame and rendered as Markdown with math (marked, DOMPurify and KaTeX,
  fetched once from the CDN and served from `/vendor/`, so the browser needs
  no internet access; without them answers show as plain text). Stop ends the
  generation after the current token.
- **Qwen2.5-Omni 7B** runs in the same resident `llm_worker.py`: the test
  script's setup (flat DRAM plan for {LM, vision, audio}, engine, params.bin)
  runs once and its weights stay on the board. Its programs depend on the
  prompt's shape (token count, media), so each message rebuilds them on the
  host (`_prepare_unified`, compile only, ~5 s for text, ~8 s with audio,
  ~14 s with an image; reported separately from the TTFT) at the same program
  addresses. Speech output (`--speak`) is not offered.
- **Five-card chat models** load ten weight shards into two engines per card.
  The runner parks its card processes between requests. Each turn resets the
  inference state and replays the conversation through the model's chat template.
  The Qwen image encoder runs on the last card and uses free memory in both DDR
  channels alongside the decoder shards. Image embeddings are reused while
  continuing the same resident conversation.
  The oldest complete turns are dropped when the prompt reaches the context limit.
  The page reports elapsed time to first token and wall-clock decode tokens/s.
- **YOLO26s-OBB** releases the resident model before running the detector on
  engine 0 of `--dev`. A completed job includes the annotated PNG, class,
  confidence and four corners in original image coordinates for each object.
- **FLUX.2 klein** runs exactly as in `flux2_klein_4b_server.py` (its module is
  imported): one `flux2_klein_4b_run_from_bin.py` process per image, or its
  resident `--serve` runner with "keep the board loaded".

Conversations:

- For Gemma 3 1B and Gemma 4 E2B/E4B, the conversation stays in the K/V cache until **New chat**. Each
  page chat has an id; a message that continues it prefills only its own
  tokens (the previous answer's unfed last token, the end-of-turn marker, the
  new user turn) at the next position, on top of the cached turns, then
  decodes. Image and audio turns join the same conversation: their soft tokens
  go into the cache like text, so a later question can refer to them.
- Context: Gemma 3 1B 512 tokens (prefill runs in chunks of up to 192 rows, each
  at its position), Gemma 4 E2B/E4B 8192. When a message would not fit, the
  conversation starts over from the latest turns (the page says how many were
  dropped). Selecting another model or FLUX unloads the chat model; coming
  back rebuilds the context from the page's text history (earlier images and
  audio are not re-encoded).
- Qwen2.5-Omni keeps the conversation by re-prefilling it each message (the
  text of the earlier turns plus this turn's image / audio, oldest turns
  dropped past the 8192-token context): its prefill has no position offset
  and the encoders share the K/V cache's DRAM, so the cache is not reused.
- Audio may be uploaded (WAV / FLAC / OGG, anything else through `ffmpeg` if
  installed) or recorded in the browser (sent as 16 kHz mono WAV; the browser
  allows the microphone on localhost or https only).
- Gemma 4's multi-engine prefill shards the prompt rows, so a first message
  shorter than the engine count ("hi") gets a one-line system preamble.
- Gemma 4's prefill program is rebuilt in-process for each message's length
  and position (a few seconds, weights stay on the board); the metrics report
  it separately and keep it out of the TTFT.
- FLUX keeps its weights on the board between images by default
  (`--reload-per-image` or unticking the box reloads per image).
- The page falls back to polling when the token stream goes silent (a proxy
  or port forward that buffers server-sent events), so answers still arrive.

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
| POST | `/api/detect` | `{image: {name, data}}` with base64 data → `{id, position}` |
| POST | `/api/generate` | FLUX: `{prompt, config, seed, no_reload}` → `{id}` |
| GET | `/api/jobs/<id>` | state, streamed `text`, `metrics` (chat) or `result` (FLUX), log |
| GET | `/api/jobs/<id>/stream` | chat: server-sent `delta` events (new text, phase, token count), then `done` with the job |
| POST | `/api/jobs/<id>/stop` | stop a chat generation |
| GET | `/api/jobs/<id>/image.png` | generated or annotated image |
| POST | `/api/release` | stop the resident process, free the board |

## Prerequisites

The models' own artifacts, built once by their test scripts:
`models/gemma3/gemma3_bin/weights_gemma3_hf.bin`,
`models/gemma4_e2b/gemma4_e2b_bin/params.bin`,
`models/gemma4_e4b/gemma4_e4b_bin/params.bin` (a chat model without them
generates them on first load, which takes minutes), and the FLUX program sets
under `models/flux2_klein_4b/flux2_klein_4b_bin/programs/`.

For the five-card models, follow the slicing and embedding steps in the
[multi-card setup guide](../apex-tp-llm/README.md#installation). Set
`APEX_TP_WORK` to the prepared workspace (default `~/apex_tp`). Qwen loads
`models/{q35_9b,q38_27b}/tp10fp4` and the tokenizer from its prepared model
directory. GPT-OSS and Gemma load `models/{gptoss20b,gemma4_26b}/tp10`.
Their runner-specific directories and tokenizer checkpoints can be selected
with `GPTOSS_WORK`, `GEMMA4_WORK`, `GPTOSS_MODEL` and `GEMMA4_MODEL`.

To enable image input, run `Q_MODEL=q35_9b python vision_prep.py` or
`Q_MODEL=q38_27b python vision_prep.py` from `models/apex-tp-llm`. Gemma uses
`python gemma4/extract_gemma4_vision.py`. These encoders also need free board
memory alongside the language model. For YOLO26, place `yolo26s-obb.pt` in
`models/apex-tp-llm/yolo26/` or set `YOLO26_WEIGHTS` to its path. The detector
uses `yolo26/env/bin/python` when installed, otherwise the server's runner
interpreter. It needs the detector's PyTorch, Ultralytics and image dependencies.
