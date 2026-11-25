# Five-card Apex inference

The unified model server runs each chat model on five Apex Kintex-7 cards with two engines and 4 GiB of DDR per card. Each engine holds a shard of every decoder layer. The FPGA engines compute attention, feed-forward layers and the language model head. The host tokenizes the prompt, coordinates the shards and streams the answer. [Unified interface and API](../unified_server/README.md) · [Card programs and memory layout](docs/TECHNICAL.md)

| Model | Server ID | Context (tokens) | Input |
|---|---|---:|---|
| Qwen3.5 9B | `q35_9b` | 16,384 | Text, image |
| Qwen3.8 27B | `q38_27b` | 8,192 | Text, image |
| GPT-OSS 20B | `gptoss20b` | 8,192 | Text |
| Gemma 4 26B-A4B | `gemma4_26b` | 8,192 | Text, image |

## Installation

Install the XDMA driver and use the firmware image expected by the checked-out runtime. Check that all five cards appear under `/dev/xdma*_user`. [Driver installation](../../README.md#installation) · [Firmware update instructions](../../README.md#7-updating-hw-bin-file)

On the HP Z420 used for bring-up, a warm boot followed by a PCIe rescan froze the host. That machine used `update_flash.py --no-boot` on each card and a cold power cycle. Its optional host tuning and XDMA service files are in `system/`. The kernel 7.0 XDMA Makefile patch uses `/home/USER/dma_ip_drivers` as a placeholder for the driver checkout.

Use Bash and a Python environment with CPU PyTorch, Transformers (including support for these model families), NumPy, safetensors, huggingface_hub and Pillow. Run the following from the repository root. Set `APEX_TP_WORK` to the directory where the generated ports and weight shards will live.

```bash
cd models/apex-tp-llm
export APEX_TP_WORK="${APEX_TP_WORK:-$HOME/apex_tp}"
python tools/download_models.py all

patches=(silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare membase cardsum dynlen ctx16k if4mix)
python port_build.py --name q35_9b --cfg9b "${patches[@]}"
python port_build.py --name q38_27b \
    --hfcfg "$APEX_TP_WORK/models/q38_27b/qwen3.5_2b_bin/Qwen3.8-27B/config.json" "${patches[@]}"
```

`port_build.py` creates a copy of the Qwen port in the workspace. GPT-OSS and Gemma also use the generated `q35_9b` engine class. To download a single checkpoint, pass `qwen3.5-9b`, `qwen3.8-27b`, `gpt-oss-20b` or `gemma-4-26b` to the download script. Set `GPTOSS_MODEL` or `GEMMA4_MODEL` before downloading to choose a different checkpoint directory. Use the same settings when preparing and serving.

### Prepare the weights

Prepare the ten decoder shards, then split each embedding table along the vocabulary dimension. Qwen 27B uses IF8 embeddings so its table and decoder fit together. Run the image preparation commands for each model that will accept images.

```bash
for model in q35_9b q38_27b; do
    APEX_Q4=fp4 Q_MODEL="$model" python extract_tp.py 10 fp4
    APEX_Q4=fp4 Q_MODEL="$model" python vision_prep.py
done
python extract_gptoss.py 10
python gemma4/extract_gemma4.py
python gemma4/extract_gemma4_vision.py

TPDIR=tp10fp4 EMB_WORK="$APEX_TP_WORK/models/q35_9b"     python tools/slice_embed.py
TPDIR=tp10fp4 EMB_WORK="$APEX_TP_WORK/models/q38_27b"    python tools/slice_embed8.py
TPDIR=tp10    EMB_WORK="${GPTOSS_WORK:-$APEX_TP_WORK/models/gptoss20b}" python tools/slice_embed.py
TPDIR=tp10    EMB_WORK="${GEMMA4_WORK:-$APEX_TP_WORK/models/gemma4_26b}" python tools/slice_embed.py
```

### Run the web interface

From `models/apex-tp-llm`, start the server on the FPGA host. Supply its network address to `--host` for access from another machine. The server picks the first five cards in PCI order. Use `--tp-devices` to choose five specific cards.

```bash
python ../unified_server/unified_server.py --host 127.0.0.1 --port 8080
```

Open the server in a browser, select a model and send a prompt. The first request loads the weights. Subsequent requests to that model reuse them. `GET /api/models` reports the available models and the resident worker.

### Run from the command line

```bash
Q_MODEL=q35_9b python run_tp.py --tp 10fp4 --engines 2 --max-context 16384 --prompt "Name the capital of France."
APEX_TP_PARAM_FACTOR=1.0 APEX_TP_PARAM_SLACK_MB=64 Q_MODEL=q38_27b \
    python run_tp.py --tp 10fp4 --engines 2 --max-context 8192 --prompt "What is 17 plus 25?"
python gptoss_tp.py --engines 2 --max-context 8192 --prompt "Why does Earth have seasons?"
GEMMA4_VISION_SHARDS=8 GEMMA4_MAX_BLOCK=288 python gemma4/gemma4_tp.py --max-context 8192 \
    --image ../../test_samples/people.jpg --soft 70 --prompt "Describe this image."
```

The unified server also supports YOLO26s-OBB on one engine. Install the detector dependencies and place the checkpoint in `yolo26/` or set `YOLO26_WEIGHTS`. [Detector setup](../unified_server/README.md#prerequisites) · [YOLO26s-OBB checkpoint](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt)

The `tools/` directory contains hardware benchmarks, per-layer reference checks and long-context tests. For an optional multi-card monitoring view, apply `aptop/aptop-multicard.patch` to the aptop revision recorded in `aptop/BASE_COMMIT`.
