# Qwen3.5-family models on several cards at once (tensor-parallel) + OllApeX chat UI

One model, all cards working on the same token at the same time. Every card holds a slice of
**every** layer, so models far larger than one card's memory run with all compute on the FPGAs:
Qwen3.5-9B and a 27B model of the same architecture on 11 Kintex-7 cards.

The folder also contains **OllApeX**, a small web chat interface on top of the runner, with
file and image upload (the vision tower runs on a card too) and a live technical panel.

Nothing in the stock `models/qwen3.5_2b` port is modified. `port_build.py` writes a patched
copy of it into a work directory.

## Measured

11 cards, hardware image `ac344ea5` (198 MHz), 4-core / 8-thread host CPU, 16 GB host RAM,
4-bit weights (FP4, one scale per 64 values), greedy decoding. The 27B figures are at context 1024.

| Model | Cards | Tokens/s | ms/token | Slowest card compute | Sync points/token |
|-------|-------|----------|----------|----------------------|-------------------|
| Qwen3.5-9B  | 11 | 6.0  | 166 | 106 ms | 65  |
| Qwen3.8-27B | 11 | 2.09 | 478 | 364 ms | 129 |

For comparison, the same 9B model as a layer pipeline over the cards (one card active at a
time) ran at 0.76 tokens/s. Loading the 27B model onto the cards takes about 2.5 minutes.

Image encoder of the 27B model (27 blocks, hidden 1152) on one card:

| Image as encoded | Patches | Image tokens | Time on the card | Program build, first use of a size |
|------------------|---------|--------------|------------------|------------------------------------|
| 256 x 256 px     | 256     | 64           | 12.7 s           | about 27 s                         |
| 384 x 256 px     | 384     | 96           | 19.5 s           | 27 to 48 s                         |
| 512 x 384 px     | 768     | 192          | 42 s             | 27 to 48 s                         |

Generation speed is the same with or without an image in the prompt. Reading the prompt costs
as much per token as generating (about 0.48 s per token for the 27B model).

Single-token decode on one card is limited by how fast weight bytes can be read (about
5.8 GB/s per engine in our measurements), so tokens/s scales roughly with
`cards x 6 GB/s / model bytes`.

## How it works

- **Slicing** (`extract_tp.py`). Per layer a card gets some value heads of the linear-attention
  layer (with the key rows they use), some query heads of the full-attention layer with their
  KV head, and a block of MLP neurons; plus a block of vocabulary rows of the LM head. A slice
  is itself a narrower layer, so the stock port's extractors quantize it unchanged.
- **Sync points** (`tp` patch). The two residual additions of a layer cannot happen on a card
  that only sees part of the layer. The card program halts there; the host adds the cards'
  partial outputs to the residual stream and writes the sum back to every card. That gives
  `2 x layers + 1` round trips per token. The cards have no link to each other.
- **LM head**. Split by vocabulary rows. Each card reports its best tokens and their logits;
  the host keeps the best overall (or samples among the cards' top four).
- **Coordination** (`run_tp.py`). One process per card, pinned to its own CPU; commands and
  results go through shared memory; each card process sleeps through the known run time of a
  segment and then polls.
- **Vision** (`vision_prep.py`, `ollapex/ollapex_vision.py`). The vision tower runs as one card
  program on the last card, in card memory the language model does not use. Head width 72 and
  MLP width 4304 are padded to multiples of 64 with zero rows and columns, which is exact.
  Image tokens carry (time, height, width) positions; the cards' rotary table rows are
  rewritten for them.

## Files

| File | Role |
|------|------|
| `port_build.py` | Builds a patched copy of the stock Qwen3.5-2B port for another model size. Each change is a named text patch that fails loudly if the stock code no longer matches. |
| `extract_tp.py` | Slices and quantizes the model weights for N cards. |
| `run_tp.py` | Command-line runner: loads the slices onto the cards, generates, prints per-segment and per-card timing. |
| `vision_prep.py` | Prepares the vision tower weights in the padded layout. |
| `ollapex/ollapex_server.py` | OllApeX HTTP server (standard library only), tokenizer, chat template, uploads. |
| `ollapex/ollapex_engine.py` | Keeps the model loaded on the cards between requests; token loop; reuse of the cards' state for follow-up turns. |
| `ollapex/ollapex_vision.py` | Image preprocessing and the card program of the image encoder. |
| `ollapex/static/` | The web page and the power calibration file. |
| `ollapex/ox_test.py`, `ollapex/ox_test_image.py` | API smoke tests (text, file, sampling; images). |
| `ollapex/ollapex.sh`, `ollapex/ollapex.service.example` | Optional systemd service and its control script. |

## Setup

Requirements: the repository's `requirements.txt` plus `safetensors` (and `pypdf` for PDF
upload in OllApeX). Tested with `transformers` 5.16.1.

All generated files live in a work directory, `$APEX_TP_WORK` (default `~/apex_tp`). The
model is selected with `Q_MODEL` (the directory name under `$APEX_TP_WORK/models`).

```bash
export APEX_TP_WORK=~/apex_tp
PATCHES="silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare"

# 1) Patched port + configuration
python port_build.py --name q35_9b --cfg9b $PATCHES
#    or, for any model of this family, from its Hugging Face config:
python port_build.py --name q38_27b --hfcfg /path/to/model/config.json $PATCHES

# 2) Put (or link) the Hugging Face model directory, with config.json, tokenizer files and
#    safetensors, under  $APEX_TP_WORK/models/<name>/qwen3.5_2b_bin/<model dir>/

# 3) Slice the weights for 11 cards (host-side preparation; about 18 min for 27B when run as
#    four parallel layer ranges, see TP_LAYERS / TP_HEAD in the script)
Q_MODEL=q35_9b  python extract_tp.py 11 v2     # -> tp11v2
Q_MODEL=q38_27b python extract_tp.py 11        # -> tp11

# 4) Run
Q_MODEL=q35_9b  python run_tp.py --tp 11v2 --nap-us 30 --prompt "Tell me about the Eiffel Tower."
Q_MODEL=q38_27b python run_tp.py --tp 11   --nap-us 30 --max-context 1024 --prompt "..."

# 5) Optional: image support for OllApeX
Q_MODEL=q38_27b python vision_prep.py
```

The number of cards must not exceed the number of value heads, and the query heads must pack
into the KV groups (`extract_tp.py` checks both). Cards are taken in PCI address order.

## OllApeX

```bash
cd ollapex
python ollapex_server.py          # http://<host>:3000
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAPEX_PORT` | `3000` | HTTP port |
| `OLLAPEX_MODEL` | `qwen3.8:27b` | Model loaded at start (`qwen3.5:9b` is also registered in `ollapex_engine.py`) |
| `OLLAPEX_CTX` | `1024` | Context length loaded onto the cards (256, 512 or 1024) |
| `OLLAPEX_AUTOLOAD` | `1` | `0`: do not load a model at start |
| `OLLAPEX_LOGDIR` | `ollapex/logs` | Per-card logs |

What it does:

- Chat with streaming output, thinking mode, system prompt, temperature / top_k / top_p / seed.
- Follow-up turns continue from the state the cards already hold, so only the new turn is read.
- Text, code and PDF upload; image upload with three detail levels (256 / 384 / 512 px long
  side, up to 64 / 144 / 256 tokens).
- Technical panel: tokens/s, time to first token, context use, per-card compute time, time per
  segment type, bytes exchanged between the CPU and the cards per token, image encoder timing.
- Power and energy panel. It is driven by the constants in `static/power.json` (wall-meter
  readings idle and generating, CPU and memory power, and a total for the cards that you
  supply). It is not a live sensor; replace the example values with readings of your system.

The server binds to all interfaces and has no authentication. It is a demo interface for a
trusted network.

Smoke tests against a running server:

```bash
python ox_test.py
OLLAPEX_EXTERNAL=1 OLLAPEX_TEST_PORT=3000 python ox_test_image.py low follow
```

## Limits

- Context up to 1024 tokens: the per-layer attention scratch of every card has to fit its
  512 MB tensor region.
- `top_k` is at most 4 (each card reports its four best logits).
- One conversation at a time; other requests wait.
- With more card processes than CPU cores the host becomes part of the token time: about
  110 ms of the 478 ms per token of the 27B model is coordination and transfer.

## Status

The numbers above were measured with these sources on the 11-card system. For this
contribution the scripts were then made host-independent (work directory from
`$APEX_TP_WORK`, English messages, debug tracing removed). That cleaned copy passes syntax
checks, and all ten patches apply to the stock port at `d0137db`, but it has not been re-run
on the cards since the cleanup.
