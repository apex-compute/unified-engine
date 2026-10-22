# apex-tp-llm

**Run large language models tensor-parallel across several Apex FPGA cards, with the OllApeX chat interface on top.**

All model compute runs on the cards. The host CPU only coordinates. It does the tokenizer and the embedding-row copy, moves data between the cards, picks the next token (sampling), and, for MoE models, picks the indices of the top-4 experts.

| Model | Cards | Context | Generation | Prompt reading |
|---|---|---|---|---|
| **GPT‑OSS‑20B** (MoE) | 5 cards × 2 engines | 8192 | **15.3 tokens/s** | 15–17 tokens/s |
| **Qwen3.5‑9B** | 5 cards × 2 engines | 16384 | **6.97 tokens/s** | 6.9 tokens/s |
| **Qwen3.8‑27B** | 5 cards × 2 engines | 8192 | **2.39 tokens/s** | 2.4 tokens/s |

<sub>HP Z420, 5 × Apex Kintex‑7 (198 MHz), 4‑bit FP4 weights, greedy decoding. Details in [Results](#results).</sub>

---

## Contents

1. [Starting point](#starting-point)
2. [Results](#results)
3. [Architecture](#architecture)
4. [What was built](#what-was-built)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Repository layout](#repository-layout)
8. [Technical details](docs/TECHNICAL.md)

---

## Starting point

| | |
|---|---|
| **Host** | HP Z420 workstation, Intel Xeon E5‑2689 (8 cores / 16 threads), 32 GB RAM |
| **Cards** | 5 × Apex Kintex‑7: four in PCIe 2.0 x4 slots, one in a PCIe 2.0 x1 slot |
| **Per card** | 2 compute engines, 2 × 2 GiB DDR3 channels, firmware `a4fba9d9`, 198 MHz |
| **Software** | Fresh Ubuntu 24.04.5, kernel 7.0; no driver, no runtime, no Python environment |
| **Goal** | Run large models with all 5 cards working on the same token, as fast as possible |

The first working setup ran Qwen3.5‑9B on the 5 cards with one engine per card at **3.2 tokens/s** (context 4096). The rest of this README describes how the system got from there to where it is now.

### PCIe layout of the cards

| Card | PCI address | Link | Attached to |
|---|---|---|---|
| 0 | `03:00.0` | Gen2 x4 | CPU root port |
| 1 | `04:00.0` | Gen2 x4 | CPU root port |
| 2 | `05:00.0` | Gen2 x4 | x16 slot |
| 3 | `06:00.0` | Gen2 x4 | chipset |
| 4 | `07:00.0` | Gen2 x1 | chipset |

---

## Results

### Speed

```mermaid
xychart-beta
    title "Generation speed (tokens/s): starting point and now"
    x-axis ["Qwen3.5-9B start", "Qwen3.5-9B", "Qwen3.8-27B", "GPT-OSS-20B first", "GPT-OSS-20B"]
    y-axis "tokens/s" 0 --> 16
    bar [3.2, 6.97, 2.39, 11.3, 15.3]
```

| Model | Weights on the cards | Context | Generation | ms/token | Prompt reading |
|---|---|---|---|---|---|
| GPT‑OSS‑20B, short reply | 11 GB | 8192 | **15.3 tokens/s** | 65.5 | 15.5 tokens/s (OllApeX: ~17) |
| GPT‑OSS‑20B, 900‑token reply | 11 GB | 8192 | 14.6 tokens/s | 68.4 | — |
| Qwen3.5‑9B | 4.5 GB | 16384 | 6.97 tokens/s | 143.6 | 6.9 tokens/s |
| Qwen3.8‑27B | 15 GB | 8192 | 2.39 tokens/s | 418 | 2.4 tokens/s |

- For GPT‑OSS‑20B the context size (2048 or 8192) does not change the speed. The cards only compute over the part of the context that is filled.
- Qwen3.5‑9B at 16K context passes long-context retrieval tests: a fact placed at the start of a 1066-token and a 3021-token prompt is found.

### GPT‑OSS‑20B: where the time of one token goes

```mermaid
pie showData
    title GPT-OSS-20B, 65.5 ms per token
    "Router + experts (24 layers)" : 39.9
    "Attention (24 layers)" : 18.6
    "LM head (201088 words)" : 6.0
    "Token start and other" : 1.0
```

### Accuracy

- Card outputs were checked step by step against a PyTorch reference that uses the same quantized weights.
- GPT‑OSS expert layer relative error: 0.5–1.1 %. Router logits: 0.3–0.5 %. The top‑4 experts the card picks match the reference.
- All 5 cards compute the router and each card picks its experts itself. The 5 picks are identical in every layer; the runner checks this on every token.

### Quantization: FP4

The cards use 4-bit **FP4 (E2M1)** with one bf16 scale per 64 values.

| Format | SQNR (27B weights) | Top‑1 accuracy, 9B (code / plain text) |
|---|---|---|
| **FP4** | 19.45 dB | **60.46 % / 97.0 %** |
| INT4 | 18.90 dB | — |
| IF4mix | 19.95 dB | 60.16 % / 95.4 % |

FP4 is the simplest format and gives the best prediction accuracy.

---

## Architecture

### Overview

```mermaid
flowchart LR
    U["Browser<br/>(OllApeX UI)"] -- HTTP --> S["OllApeX server<br/>tokenizer · chat template"]
    S -- pipe --> D["Driver process<br/>token loop · sampling"]
    D <-- "shared memory<br/>(command counter, shares)" --> W0["Card process 0"]
    D <--> W1["Card process 1"]
    D <--> W2["Card process 2"]
    D <--> W3["Card process 3"]
    D <--> W4["Card process 4"]
    W0 -- "PCIe (xdma)" --> C0["Card 0<br/>engine 0 · DDR0<br/>engine 1 · DDR1"]
    W1 --> C1["Card 1"]
    W2 --> C2["Card 2"]
    W3 --> C3["Card 3"]
    W4 --> C4["Card 4"]
    A["aptop<br/>(monitor)"] -. state file .- D
```

- **Tensor parallelism:** Every card holds a slice of every layer, and all 5 cards work on the same token at the same time. Each card has two engines and each engine is one slice, so there are 10 slices.
- **Sync points:** A layer's output is the sum of all slices' shares. Each card returns its share (the sum of its two engines), and the host hands all shares to every card. The cards do the adding, so the residual stream stays on the cards.
- **There is no link between the cards.** Every exchange goes over PCIe through host memory. Keeping the fixed cost per sync point low is the key to speed.

### Processes and CPU placement

| CPU | Runs | Note |
|---|---|---|
| 0–7 | xdma completion threads | Driver in poll mode, each thread bound to one core |
| 8 | OllApeX driver process (token loop) | First entry of `APEX_TP_CPUS` |
| 9–13 | 5 card processes (one per card) | Each drives both engines of its card |

The card processes open their DMA channels once and do small transfers with direct `pwrite`/`pread` calls. One small transfer takes ~10 µs on an x4 card and ~13 µs on the x1 card.

### Inside a card: two engines, two memory channels

```mermaid
flowchart TB
    subgraph K["Apex Kintex-7 card"]
        direction LR
        subgraph E0["Engine 0 · slice 2c"]
            P0["Weights (FP4)"]
            T0["Tensors · KV cache"]
            G0["Program (top 64 MB)"]
        end
        subgraph E1["Engine 1 · slice 2c+1"]
            P1["Weights (FP4)"]
            T1["Tensors · KV cache"]
            G1["Program (top 64 MB)"]
        end
        E0 --- D0[("DDR3 channel 0<br/>2 GiB")]
        E1 --- D1[("DDR3 channel 1<br/>2 GiB")]
    end
```

Each engine uses its own DDR channel. One engine reads weights at ~5.7 GB/s, and two engines on separate channels reach ~11.4 GB/s together, so the second engine almost doubles the bandwidth.

### How GPT‑OSS‑20B is split across the cards

GPT‑OSS‑20B has:
- 24 layers, hidden size 2880.
- 64 query heads and 8 KV heads.
- 32 experts, 4 active per token.
- Sliding-window (128) and full-attention layers, alternating.
- Attention sinks and YaRN RoPE.
- A vocabulary of 201088 tokens.

| Slice | Card / engine | Query heads | Expert neurons (for each of the 32 experts) | LM head rows |
|---|---|---|---|---|
| 0 | 0 / 0 | 4 | 320 | 20160 |
| 1 | 0 / 1 | 4 | 320 | 20160 |
| 2 | 1 / 0 | 4 | 320 | 20096 |
| 3 | 1 / 1 | 4 | 320 | 20096 |
| 4 | 2 / 0 | 8 | 320 | 20096 |
| 5–9 | 2/1 … 4/1 | 8 | 256 | 20096 |

The neurons of every expert are split over the 10 slices. Whichever 4 experts are picked, every engine does the same amount of work, and no expert ever has to be moved between cards. The whole model (~11 GB) stays resident in card memory.

### GPT‑OSS‑20B: one token

```mermaid
sequenceDiagram
    autonumber
    participant D as Driver (host)
    participant W as Card processes (×5)
    participant C as Cards (×5, 10 engines)
    D->>W: token start (embedding row)
    W->>C: one DMA: embedding + mask templates + position registers
    loop 24 layers
        W->>C: A: attention (each engine its own heads)
        C-->>W: engine 1's share added on engine 0 · card share read
        D->>W: shares to all cards
        W->>C: R: add shares · RMSNorm · router (32 logits)
        C-->>W: 32 router logits (64 bytes)
        W->>W: top-4 expert pick (indices only)
        W->>C: one DMA: 4 logits + 4 subroutine addresses
        W->>C: E: 4 expert subroutines (jump) · weighted sum
        C-->>W: card share read
        D->>W: shares to all cards
    end
    W->>C: H: final norm · LM head (each engine its vocabulary rows)
    C-->>W: each engine's best 4 tokens (hardware argmax)
    W-->>D: candidates
    D->>D: pick the next token (greedy / sampling)
```

There are **49 sync points per token**: attention and router+experts for every layer, plus the LM head.

### Attention: all heads in one matrix multiplication

```mermaid
flowchart LR
    q["q (64 per head)"] --> R["RoPE"] --> QP["Q' row<br/>[ rope(q) | one-hot(head) ]<br/>128 wide"]
    k["k · 1/8"] --> RK["RoPE"] --> KP["K' cache<br/>row 0 = [ 0 | sinks ]<br/>row t = [ k_t | 0 ]"]
    QP --> QK["Q'·K'ᵀ + mask<br/>softmax<br/>(one matmul, all heads,<br/>length from a register)"]
    KP --> QK
    QK --> SC["Probabilities<br/>[heads × length]"]
    v["v"] --> VT["Vᵀ (packed)<br/>every token: one 64-row block<br/>every 64 tokens: rebuilt"]
    SC --> PV["P·V<br/>(one matmul, K = length)"]
    VT --> PV
    PV --> O["o projection → share"]
```

- **Sinks:** The second half of a Q' row is the head's one-hot vector, and row 0 of the K' cache holds the sink values of all heads. The sink column therefore gets each head's own value, and one mask row serves every head.
- **Scale:** The 1/√64 = 1/8 scale is folded into the K weights and bias. It is a power of two, so this is exact.
- **Length:** The card's length register (a multiple of 64) sets the column count of QK and the K dimension of P·V. Empty context is never computed.

---

## What was built

### 1. Host and driver setup

| Work | Result |
|---|---|
| Built the Xilinx xdma driver for kernel 7.0 ([patch](system/xdma-makefile.patch)) | All 5 cards visible; the driver loads at boot |
| Put xdma in **poll mode** (`poll_mode=1`) | Polling instead of interrupts: lower latency for small transfers |
| Added the `apex-tune` service: performance governor, deep C-states (C3+) off | No CPU wake-up latency |
| CPU placement: card processes on 9–13, xdma threads on 0–7 | The processes do not get in each other's way |
| Persistent DMA files + direct `pwrite`/`pread` | ~10 µs per small transfer |
| Embedding table kept in RAM | No disk access per token |
| systemd services `xdma` → `apex-tune` → `ollapex` | Everything starts by itself at boot, in order |

### 2. Tensor-parallel runtime (Qwen3.5 family)

The vendor's Qwen3.5 port is left unchanged. `port_build.py` writes a patched copy of it into a work directory.

| Technique | Effect |
|---|---|
| **Slicing** (`extract_tp.py`): every card gets a slice of the heads and neurons of every layer | Models larger than one card's memory run |
| **Summing on the cards** (`cardsum`): the cards add the shares; the residual stays on the card | The host only moves bytes |
| **Two engines, two DDR channels**: one slice per engine, each in its own memory channel | 9B: 235 → 171 ms/token |
| **Shared parts buffer + on-card pre-sum** (`presum`): engine 1's share is added on engine 0 | One read per card: 171 → 143.6 ms/token |
| **Dynamic length** (`dynlen`): attention length comes from a card register | Empty context is not computed: 311 → 235 ms/token |
| **16K context** (`ctx16k`): one copy per KV head, persistent chunked Vᵀ | 9B runs with 16384 context |
| **Memory layout** (`membase`): separate parameter / tensor / program regions | The 27B weights and a long context fit together |
| **FP4 quantization** | Best prediction accuracy, simplest format |

```mermaid
xychart-beta
    title "Qwen3.5-9B: ms/token (lower is better)"
    x-axis ["First setup", "dynlen", "2 engines", "shared buffer", "presum + RAM embedding"]
    y-axis "ms/token" 0 --> 320
    bar [311, 235, 171, 153.5, 143.6]
```

### 3. GPT‑OSS‑20B (MoE) port

The card program for the GPT‑OSS architecture was written from scratch (`gptoss_card.py`, `gptoss_tp.py`, `extract_gptoss.py`).

| Component | Solution |
|---|---|
| MXFP4 expert weights | Unpacked with transformers' own converter, requantized to the card's FP4 |
| Attention sinks | Row 0 of the K' cache + the one-hot half of Q' |
| Sliding window (128) | The card updates the mask row on every token from two template copies |
| YaRN RoPE | transformers' own YaRN tables, with the same scaling |
| Clamped SwiGLU (GELU‑sigmoid, clamp 7) | The clamp runs inside the GEMV; GELU is an identity matmul on 64×64 blocks |
| Expert selection | The router runs on the card; each card process picks the top‑4 **indices** from its own card's 32 logits |
| Expert call | Every (layer, expert) is a precompiled subroutine; the card **jumps through a register** to the 4 picked ones (`jump_reg_abs`) |
| Expert weights | The softmax over the 4 logits runs on the card |
| Token start | Embedding, mask templates and 9 position registers are written with **one DMA** |

**Speed gains** (each row measured against the step before it, under the same conditions):

| Optimization | Gain (ms/token) |
|---|---|
| One DMA for token start and for the expert pick (instead of separate small programs) | −11.5 |
| Persistent DMA files, direct register access, no numpy on the hot host path | −10.9 |
| Expert pick done locally in each card process, `struct`‑based | −3.9 |
| GELU on 64×64 blocks (45 → 8.6 µs per expert) | −4.5 |
| P·V only over the filled context | −3.3 |
| QK + softmax and P·V of all heads in one matmul each | −3.1 |
| **Overall** | **11.3 → 15.3 tokens/s** |

On an 8-head slice the card time of attention went from 691 µs to 435 µs per layer, and the expert segment from 1237 µs to 1083 µs.

### 4. OllApeX chat interface

- **Web UI (port 3000):**
  - Model and context choice (256 – 16384).
  - File attachments.
  - Image input for the Qwen models; the vision encoder runs on a card.
  - Live technical panel: cards, links, speed, power estimate.
- **gpt-oss:20b:** harmony chat template. The model's analysis is shown as "thinking" and its answer as normal text. The "think" switch sets the reasoning effort (low / medium).
- **Math:** formulas in replies (`\[ … \]`, `$$ … $$`, `\( … \)`, `$ … $`) are typeset with [KaTeX](https://katex.org) 0.16.11, served from `ollapex/static/katex` (MIT license), so no internet access is needed. Markdown tables and horizontal rules are rendered too.
- **Sampling:** greedy or temperature / top‑k / top‑p. Every engine reports its best 4 tokens through the hardware argmax, and the host picks among the overall best candidates.
- Replies are limited to 1024 tokens by default.
- For the Qwen models, the part of the previous conversation already held on the cards is reused; only the new message is read.
- The LM head is skipped while the prompt is read, so GPT‑OSS reads prompts at ~17 tokens/s.
- Between requests the card processes park and use no CPU.
- Runs as a systemd service: `ollapex.sh start | stop | restart | status | log`.

### 5. aptop: multi-card monitoring

Patch for [aptop](https://github.com/celikomer/aptop), an nvtop-like monitor for Apex cards ([aptop/aptop-multicard.patch](aptop/aptop-multicard.patch)):

- Detects how many cards are installed and lays them out in a grid of up to 6 columns.
- OllApeX writes the state file aptop reads (`aptop-workload/v1`): the loaded model, the phase (loading / idle / prompt reading / generating), and per-card memory use (weights, KV, program).

### 6. Measurement and test tools

| Tool | What it measures |
|---|---|
| `tools/test_gptoss.py` | GPT‑OSS layer segments against a PyTorch reference (A / R / E error, top‑4 match) |
| `tools/prof_gptoss.py` | Times every card step of one layer separately |
| `tools/test_ctx16k.py`, `tools/test_dynlen.py`, `tools/needle.sh` | Qwen long-context correctness and retrieval test |
| `tools/bench_pcie.py`, `tools/bench_dmalat.py`, `tools/bench_dmaconc.py` | PCIe/DMA latency, one card and all 5 at once |
| `tools/bench_2eng.py` | Memory bandwidth of the two engines |
| `tools/quant_compare.py` | SQNR of the quantization formats |
| `tools/download_models.py` | Downloads the Hugging Face checkpoints into the places the slicing scripts read |
| `run_tp.py --eval-file` | Teacher-forced top‑1 accuracy on the cards |

---

## Installation

### Requirements

- Ubuntu 24.04, `build-essential`, kernel headers.
- A Python 3.12 virtual environment with `torch` (CPU), `transformers` ≥ 5.20, `numpy` 1.26, `safetensors`.
- The vendor runtime [apex-compute/unified-engine](https://github.com/apex-compute/unified-engine), at a version that matches firmware `a4fba9d9`.

> On this Xeon, PyTorch's multi-threaded paths can use a CPU instruction the processor lacks. All scripts run with `torch.set_num_threads(1)`.

### 1. Get the code

The branch `5-Apex-fpga` is the vendor runtime (matching firmware `a4fba9d9`) with this project in `models/apex-tp-llm/`:

```bash
git clone -b 5-Apex-fpga https://github.com/dogankumova/unified-engine.git ~/unified-engine
```

### 2. xdma driver and host tuning

```bash
git clone https://github.com/Xilinx/dma_ip_drivers.git ~/dma_ip_drivers
cd ~/dma_ip_drivers/XDMA/linux-kernel/xdma
sed "s#/home/USER#$HOME#" ~/unified-engine/models/apex-tp-llm/system/xdma-makefile.patch | patch Makefile
make && sudo mkdir -p /lib/modules/$(uname -r)/xdma && sudo cp xdma.ko /lib/modules/$(uname -r)/xdma/

sudo cp ~/unified-engine/models/apex-tp-llm/system/{xdma,apex-tune}.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now apex-tune xdma
ls /dev/xdma*_h2c_0          # one line per card
```

### 3. Patched model port

```bash
cd ~/unified-engine/models/apex-tp-llm
export APEX_TP_WORK=~/apex_tp APEX_TP_CPUS=8,9,10,11,12,13

# patched Qwen port (GPT-OSS also builds on this port's engine class)
python port_build.py --name q35_9b --cfg9b silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare membase cardsum dynlen ctx16k if4mix
python port_build.py --name q38_27b --hfcfg <Qwen3.8-27B dir>/config.json silu kpad attn1 noeye heads shard recgrp tp abfuse vtshare membase cardsum dynlen ctx16k if4mix
```

### 4. Models and slicing

Place the Hugging Face model directories under `$APEX_TP_WORK/models/<name>/qwen3.5_2b_bin/<Model>/` (a symlink is enough). For GPT‑OSS, point `GPTOSS_MODEL` at the model directory.

```bash
python tools/download_models.py all            # the Hugging Face checkpoints, into the places below

# Qwen: 10 slices (5 cards x 2 engines), FP4
APEX_Q4=fp4 Q_MODEL=q35_9b  python extract_tp.py 10 fp4      # -> $APEX_TP_WORK/models/q35_9b/tp10fp4
APEX_Q4=fp4 Q_MODEL=q38_27b python extract_tp.py 10 fp4      # -> $APEX_TP_WORK/models/q38_27b/tp10fp4

# GPT-OSS-20B: 10 slices; can run as parallel copies over layer ranges (GPTOSS_LAYERS=0:6 ...)
GPTOSS_MODEL=<gpt-oss-20b dir> python extract_gptoss.py 10   # -> $APEX_TP_WORK/models/gptoss20b/tp10
```

### 5. OllApeX service

```bash
sed "s/USER/$USER/g" system/ollapex.service | sudo tee /etc/systemd/system/ollapex.service
sudo systemctl daemon-reload && sudo systemctl enable --now ollapex
ollapex/ollapex.sh status
```

The interface is at `http://<host>:3000`.

---

## Usage

### Web interface

Pick **gpt-oss:20b**, **qwen3.5:9b** or **qwen3.8:27b** from the model menu. A model is loaded onto the cards the first time it is chosen: about 40 s for GPT‑OSS and about 1.5 minutes for the 27B model.

### API

```bash
curl -X POST localhost:3000/api/load -d '{"model": "gpt-oss:20b", "num_ctx": 8192}'
curl localhost:3000/api/status
curl -N -X POST localhost:3000/api/chat -d '{
  "model": "gpt-oss:20b",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "think": false,
  "options": {"temperature": 0.7, "top_k": 4, "num_predict": 1024}
}'
```

`/api/chat` streams newline-delimited JSON events: `status`, `thinking`, `content`, and `done` (with speed and timing statistics).

### Command line

```bash
# GPT-OSS-20B
python gptoss_tp.py --engines 2 --max-context 8192 --max-new-tokens 200 --prompt "..."
GPTOSS_PROF=1 python gptoss_tp.py ...        # step-by-step time breakdown per card

# Qwen
Q_MODEL=q35_9b python run_tp.py --tp 10fp4 --engines 2 --max-context 16384 --prompt "..."
Q_MODEL=q35_9b python run_tp.py --tp 10fp4 --engines 2 --eval-file text.txt --eval-tokens 1000
```

---

## Repository layout

```
models/apex-tp-llm/           (in unified-engine, branch 5-Apex-fpga)
├── port_build.py             builds the patched copy of the Qwen3.5 port
├── extract_tp.py             Qwen slicing
├── run_tp.py                 Qwen tensor-parallel runner
├── vision_prep.py            Qwen vision encoder weights
├── gptoss_card.py            GPT-OSS card program (one engine = one slice)
├── gptoss_tp.py              GPT-OSS runner and card processes
├── extract_gptoss.py         GPT-OSS slicing (MXFP4 → FP4)
├── ollapex/                  chat interface: server, engine, GPT-OSS backend, web page
├── system/                   systemd services and the xdma Makefile patch
├── aptop/                    aptop multi-card patch
├── tools/                    tests, benchmarks and profilers
└── docs/TECHNICAL.md         card program, register map, measurements
```
