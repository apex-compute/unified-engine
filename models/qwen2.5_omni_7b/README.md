# Qwen2.5-Omni-7B

Qwen2.5-Omni-7B on the Apex Compute Unified Engine: text, image and audio in;
text and speech out. Every learned matrix operation runs on eight FPGA
engines:
- the Thinker LLM with its vision and audio encoders;
- the Talker that turns the reply into speech codec tokens;
- Token2Wav, the DiT and BigVGAN that turn codec tokens into a waveform.

The host does control, DMA, tokenization, media decode and preprocessing, the
BF16 embedding gather, and the final compare of the eight per-engine argmax
candidates.

- [What it does](#what-it-does)
- [Supported hardware](#supported-hardware)
- [Design](#design)
- [Setup](#setup)
- [Running](#running)
- [Limits](#limits)
- [Performance with the default prompts](#performance-with-the-default-prompts)
- [Benchmark tiers](#benchmark-tiers)
- [Known issues](#known-issues)
- [Reports and further reading](#reports-and-further-reading)

## What it does

| Input | Output | Flags |
| :--- | :--- | :--- |
| text | text | `--prompt "..."` (or none, for the default prompt) |
| image + text | text | `--image [PATH]` (default `test_samples/yosemite.jpg`) |
| audio + text | text | `--audio [PATH]` (default `test_samples/apex.wav`) |
| image + audio + text | text | `--image --audio` |
| any of the above | text **and speech** (24 kHz WAV) | add `--speak [Chelsie\|Ethan]` |

- **Multi-camera input:** `--frames N` encodes N camera frames in one run.
- **Image resolution:** `--vision-res medium` uses 896×896 images (1024 soft tokens) instead of the default 336×336 (144).
- **Generation:** greedy, up to `--max-new-tokens` (default 128). It stops on `<|im_end|>`.

## Supported hardware

The board is identified from its HW_INFO register: core count, DRAM size, AXI
width and clock. The model always uses engines 0–7; on a 12-engine image,
engines 8–11 stay idle. A board it has no memory map for is refused before
anything is allocated.

| Board | Image (verified here) | Engines used | HBM | Engine clock | 8-engine peak |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Alveo U50 | `0x25511139` | 8 of 8 | 8 GiB | 366.7 MHz | 375.5 GFLOPS |
| Alveo U55C | `0x25511139` | 8 of 12 | 16 GiB | 290.0 MHz | 297.0 GFLOPS |

Both bitstreams need non-aligned DMA support. The runner never reprograms the
FPGA.

## Design

**Stages.** Every stage runs on all eight engines:

| Stage | What it does | Weights |
| :--- | :--- | :--- |
| Vision encoder | ViT over 14×14 patches, 2×2 merge to soft tokens | IF4 |
| Audio encoder | conv front end + transformer over 128-bin log-mel, 100 frames/s | IF4 |
| Thinker prefill | 28-layer Qwen2 decoder over the whole prompt (text + soft tokens), tensor-parallel | IF4 Q/K/O/gate/up/down, BF16 V |
| Thinker decode | one token per step; KV cache resident in each engine's window | IF4, column-sharded per engine |
| Talker (`--speak`) | 24-layer decoder: Thinker hidden states → codec tokens | IF4 |
| Token2Wav (`--speak`) | flow-matching DiT (22 blocks, 36 evaluations) + BigVGAN vocoder → 24 kHz audio | BF16 |

**Sharding.**
- **Projection weights:** every projection is split into 64-aligned column shards, one per engine, staged once and shared by prefill and decode.
- **Attention:** decode splits it across KV groups.
- **LM head:** column-sharded. Each engine's argmax register holds its shard's winner, and the host reads back just those eight values (16 bytes) to pick the token.
- **Embeddings:** the 152064 × 3584 table stays BF16 in host RAM, and the host gathers rows and DMAs them in.

**DRAM layouts** (`--layout`):
- **`windowed` (default):** eight private 1 GiB windows, one per engine. Each window holds that engine's weight shards, a pool shared over time by the stage currently running, private scratch, a guard, and its program (ISA) space. The window bases come from the board:
  - **U50:** one 1 GiB HBM switch region per engine.
  - **U55C:** one whole memory controller per engine. On the 16 GiB image the windows alternate between the two HBM stacks (0, 9, 2, 11, 4, 13, 6, 15 GiB).

  Programs are recompiled each run. Token2Wav for `--speak` is built and uploaded at run time.
- **`flat`:** one consecutive 8 GiB range in which every stage's weights and programs stay resident. It makes `--run_from_bin` possible (no compile on repeat runs), at the cost of the engines contending for the same memory (see [Known issues](#known-issues)). On the 16 GiB U55C the flat map still uses only the first 8 GiB.

**Cached artifacts** (in `qwen2.5_omni_7b_bin/`, all regenerated automatically):
- **`Qwen2.5-Omni-7B/`:** the Hugging Face checkpoint, pinned to revision `ae9e1690…` in `qwen2.5_omni_7b_config.json`.
- **`params.bin` + `params.json`:** every learned tensor in its on-device format, with a validated manifest. The Talker and Token2Wav regions are appended on first `--speak`.
- **`programs.bin` + `programs.json`:** the compiled eight-engine programs, fingerprinted against the hardware, compiler and params.
- **`processor/`:** the tokenizer and media-processor metadata.

## Setup

```bash
python3 -m venv ~/venv && ~/venv/bin/pip install -r requirements.txt   # repo root
source ~/venv/bin/activate
```

Nothing has to be copied by hand. Everything the run needs is downloaded or
rebuilt on first use, at the pinned checkpoint revision:

| Run | Downloads (once) | Builds (once) |
| :--- | :--- | :--- |
| any | Thinker shards 1–4 (~20 GB) + metadata | `params.bin` Thinker/vision/audio regions, `processor/` |
| `--speak` | shard 5 (talker/token2wav, 2.4 GB) + `spk_dict.pt` | Talker region (flat layout: also Token2Wav region) |

- **Stale caches are rebuilt automatically:** a `params.bin` from an older schema, another config or an interrupted write is rebuilt, and a stale `programs.bin` is discarded and recompiled.
- **Speech files are only fetched for `--speak`:** runs without it never download shard 5 or `spk_dict.pt`.
- **Converting separately:** to do the one-time conversion on its own, run `python models/qwen2.5_omni_7b/qwen2.5_omni_7b_weights.py`.

## Running

Run from the repository root. `--multi-core 8` is required.

```bash
# Default prompts (text / image / audio / image+audio)
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --image
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --audio
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --image --audio

# Your own prompt and media
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 \
  --image /path/to/photo.jpg --prompt "Describe this image in detail."

# Speech output (FPGA Talker + FPGA Token2Wav); writes a .wav next to the summary
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --speak

# Per-phase FPGA profile instead of generation
python models/qwen2.5_omni_7b/qwen2.5_omni_7b_test.py --multi-core 8 --image --profile

# Benchmark tiers (see benchmark.md)
python models/qwen2.5_omni_7b/benchmark.py --multi-core 8 --low      # or --medium / --high

# Automated hardware checks (text, image, audio, image+audio)
python model_auto_test.py --only qwen2.5_omni_7b qwen2.5_omni_7b_vlm \
  qwen2.5_omni_7b_audio qwen2.5_omni_7b_joint --verbose
```

The default prompts are:

| Input | Default prompt |
| :--- | :--- |
| text | "Explain why the sky is blue in one sentence." |
| image | "Describe the picture in detail." |
| audio | "Transcribe the speech exactly." |
| image + audio | "First transcribe the audio. Then briefly describe the image." |

Each run writes a Markdown summary next to the script, named after the CLI
config, unless you pass `--summary PATH` or `--no-summary`. For each stage it
reports work (issued and model GFLOP), HW-counter latency, throughput, percent
of peak and CPU wall time, then TTFT, decode speed, and the speech stages for
`--speak`. HW latency is wrap-corrected: the 32-bit counter wraps every 187 s
at 366.7 MHz and every 237 s at 290 MHz.

## Limits

| Limit | Value |
| :--- | :--- |
| Context (prompt + generation) | 8192 tokens (windowed); flat: largest that fits, 7680 with speech |
| Context with FPGA `--speak` (windowed) | 6144 tokens |
| Image | 336×336 → 144 soft tokens (default) or 896×896 → 1024 soft tokens; any input is resized |
| Audio | at most 600 mel frames (6 s) → up to 150 soft tokens; stereo/any rate is mixed and resampled |
| Speech output | capped at 300 codec tokens (6.00 s); longer replies are cut |
| Speakers | Chelsie (default), Ethan |
| Decoding | greedy only |

## Performance with the default prompts

Every run below uses the default prompt and the default (windowed) layout. On
each board the runs used the same code. All runs produced correct output:
- **Text:** a correct one-sentence answer about Rayleigh scattering.
- **Image:** a detailed Yosemite sunset description.
- **Audio:** the exact transcription "unified engine apex compute custom ml inference accelerator".

Times are HW counters, and "% peak" is the run's issued throughput.

| Run (prompt tokens) | Stage | U50 @ 366.7 MHz | U55C @ 290 MHz | Ratio |
| :--- | :--- | ---: | ---: | ---: |
| text (29) | prefill | 1159 ms (87.2%) | 1464 ms (87.2%) | 1.26 |
| | decode, 39 tokens | 19.92 tok/s (75.2%) | 15.94 tok/s (76.1%) | 0.80× speed |
| image (170) | vision, 144 soft tokens | 2937 ms (68.8%) | 3693 ms (69.2%) | 1.26 |
| | prefill | 6597 ms (90.1%) | 8348 ms (90.1%) | 1.27 |
| | TTFT | 9.53 s | 12.04 s | 1.26 |
| | decode, 128 tokens | 18.61 tok/s (70.6%) | 14.87 tok/s (71.4%) | 0.80× speed |
| audio (154) | audio, 128 soft tokens | 1186 ms (75.8%) | 1484 ms (76.7%) | 1.25 |
| | prefill | 6004 ms (89.7%) | 7583 ms (89.8%) | 1.26 |
| | TTFT | 7.19 s | 9.07 s | 1.26 |
| | decode, 9 tokens | 19.08 tok/s (72.3%) | 15.27 tok/s (73.1%) | 0.80× speed |
| image + audio (306) | vision / audio | 2938 / 1180 ms | 3692 / 1482 ms | 1.26 / 1.26 |
| | prefill | 11870 ms (90.5%) | 15019 ms (90.5%) | 1.27 |
| | TTFT | 15.99 s | 20.19 s | 1.26 |
| | decode, 98 tokens | 17.14 tok/s (65.3%) | 14.27 tok/s (68.7%) | 0.83× speed |

Both boards produced word-for-word identical output for every prompt.

Every stage scales with the engine clock (366.7 / 290 = 1.26×) and runs at
the same fraction of peak on both boards. The vision, audio and prefill stages
are 1.25–1.27× slower on the U55C, and short-context decode runs at 0.80–0.83×
the speed. At long context decode loses less than the clock, because more of
each step is memory time: Medium tier decode is 0.93× (below).

**Speech** (`--speak`, text prompt, 300 codec tokens = 6.00 s of audio, speaker
Chelsie). On the U50 this comes from the earlier `0xbaa4c63b` reports at the
same 366.7 MHz clock; the current U50 image hangs in Token2Wav (see
[Known issues](#known-issues)).

| Stage | U50 @ 366.7 MHz | U55C @ 290 MHz | Ratio |
| :--- | ---: | ---: | ---: |
| Talker (prefix + decode) | 8.79 s, 39.0 codec tok/s, 29.3% peak | 11.00 s, 31.0 codec tok/s, 29.4% peak | 1.25 |
| Token2Wav DiT (36 evaluations) | 62.05 s, 69.0% peak | 78.31 s, 69.1% peak | 1.26 |
| Token2Wav BigVGAN | 8.91 s, 33.9% peak | 10.99 s, 34.7% peak | 1.23 |
| **Speech total** | **79.75 s (0.075× real time)** | **100.30 s (0.06× real time)** | **1.26** |

Speech is not real time: 6 s of audio takes 80–100 s of device time, most of
it in the DiT. The Talker alone decodes at 0.6–0.8× of the 50 codec-tokens/s
real-time rate.

## Benchmark tiers

`benchmark.py` runs three fixed workloads end to end. The U50 numbers come from
[`benchmark.md`](benchmark.md); the U55C numbers were measured on aztec.

| Tier | Input | LM prefill rows |
| :--- | :--- | ---: |
| Low (voice command) | 2.55 s audio | 849 |
| Medium (single camera) | 1 × 896×896 image + 5.10 s audio | 2047 |
| High (multi-camera) | 4 × 896×896 images + 6.00 s audio | 6144 |

| Tier | Stage | U50 @ 366.7 MHz ([`benchmark.md`](benchmark.md)) | U55C @ 290 MHz | Ratio |
| :--- | :--- | ---: | ---: | ---: |
| Low | audio / prefill | 0.61 / 33.48 s | 0.77 / 42.27 s (90.8%) | 1.26 / 1.26 |
| | TTFT | 34.09 s | 43.04 s | 1.26 |
| | decode | 14.90 tok/s | 12.22 tok/s | 0.82× speed |
| Medium | vision / audio / prefill | 22.27 / 1.18 / 84.51 s | 27.46 / 1.48 / 106.22 s (90.1%) | 1.23 / 1.26 / 1.26 |
| | TTFT | 107.96 s | 135.16 s | 1.25 |
| | decode | 9.75 tok/s | 9.02 tok/s | 0.93× speed |
| High | vision / audio | 90.15 / 1.45 s | 109.88 / 1.76 s | 1.22 / 1.22 |
| | prefill | 339.06 s *(extrapolated)* | **369.26 s (87.1%)** | — |
| | TTFT | 430.65 s *(extrapolated)* | **480.90 s** | — |
| | decode (6144-token context) | 7.05 tok/s *(extrapolated)* | **4.89 tok/s** | — |

- **Low and Medium** scale with the clock, like the default prompts.
- **High** has only been measured on the U55C. The U50 High row is extrapolated, because the current U50 image hangs on that load (see [Known issues](#known-issues)), so the two can't be compared yet. It ran with correct output: the transcription, after which the model ended its reply (19 tokens).

## Known issues

- **U50 image `0x25511139` hangs under long, heavy loads:** Token2Wav (`--speak`), and the Medium/High tiers. It runs the engine at 366.67 MHz, where the U50 build misses timing badly (about 200k failing endpoints). A local build of the same RTL at 333.33 MHz passed the 36-evaluation Token2Wav test that this image fails. Until the U50 image is rebuilt at 333.33 MHz, run speech and the large tiers on the U55C. The U50's High tier in `benchmark.md` is extrapolated for this reason.
- **The flat layout is slower on the U55C:** with every stage resident, the engines contend for the same memory controllers. Thinker decode drops from 75% to 43% of peak, and the Talker from 29% to 23%, against the windowed layout. Use `--layout flat` only when you need `--run_from_bin`.
- **Thermal headroom:** on p2's U50 the die reaches 100 °C after about 5 minutes of full load, which is the limit for the High tier there. The U55C on aztec stays below 75 °C.

## Reports and further reading

- [`benchmark.md`](benchmark.md): the U50 tier benchmark (TTFT, stage throughput, decode, speech).
- [`vision_pad_removal.md`](vision_pad_removal.md): the vision padding removal, which needs non-aligned DMA.
- `benchmark.py`: the module docstring explains the tier shapes and the `--standalone` cross-check.
- `qwen2.5_omni_7b_layout.py`: both DRAM maps, with the per-board reasoning.
- `multi_engine_shard.py` (repo root): `tiled_window_bases()`, the board-specific HBM window maps.
