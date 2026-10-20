# Moonshine-base (ASR)

Bring-up of [moonshine-ai/moonshine-base](https://huggingface.co/moonshine-ai/moonshine-base)
(61M, English, MIT). Encoder-decoder transformer on raw 16 kHz PCM via a 3-layer
conv stem (no mel/STFT). Weights come from HF `transformers`-format safetensors;
the upstream `moonshine` pip package is not used.

## Layout

- **moonshine_test.py** – the ACCELERATOR path (UnifiedEngine): full forward pass
  (on-device im2col conv stem + GroupNorm, 8 encoder layers, cross-K/V, 194 fixed
  decode-step programs with K/V caches, HW argmax). Compiles ONCE at the
  `max_audio_seconds` cap geometry; per-clip inputs are data only (padded PCM, frame
  masks, GroupNorm count). First run compiles (~85 s) and freezes
  `moonshine_bin/{params,programs}.{bin,json}`; later runs replay the image (no weight
  upload, no compile) when its signature (code/config sha1, weights, cap, DRAM bases)
  matches, otherwise they recompile. HW-proven 2026-10-20 single-engine: exact
  transcript match, ~0.8 s on hardware for 5 s of audio. `--engines N` shards the stem
  (row-split, per-engine GroupNorm partial sums) and the encoder (head-split attention,
  row-split LN/MLP/cross-K/V) and the decoder (head-split attention, N-split fc1 and
  LM head with host-compared argmax candidates) over N engines via `multi_engine_shard`.
  Decode is ONE body program for all 194 steps (runtime kv-length + cache-row registers,
  llama-style dispatch table), so compile is seconds instead of ~85 s. With `--engines N`
  the model map moves to the second HBM stack and every decode projection is materialized
  as controller-private column shards (gemma3 scheme, `shards.bin`), so the M=1 weight
  stream is spread over N memory controllers instead of contending on one.
- **moonshine_cpu_reference.py** – CPU/CUDA reference, no hardware: flat op-by-op port
  of HF `MoonshineForConditionalGeneration` (conv stem, 8+8 layers, partial RoPE,
  cross-attn, gated MLP, greedy decode with K/V cache) plus the host plumbing both
  paths share (HF download into `moonshine_bin/`, tokenizer, audio, `--validate`,
  `--dump-golden`). Validated bit-exact vs HF (inf dB encoder, all tokens match).
- **moonshine_config.json** – shapes, padded shapes, ops flags, `hw_notes`, DRAM bases.
- **moonshine_bin/** – downloaded weights + golden (generated, not committed).

## Prerequisites

```bash
pip install -r models/moonshine/moonshine_requirements.txt
```

## Usage (repo root)

```bash
# CPU/CUDA reference (no board)
python models/moonshine/moonshine_cpu_reference.py                    # apex.wav
python models/moonshine/moonshine_cpu_reference.py --device cpu --audio x.wav
python models/moonshine/moonshine_cpu_reference.py --validate         # exit 0 iff tokens == HF
python models/moonshine/moonshine_cpu_reference.py --dump-golden      # moonshine_bin/cpu_golden.npz

# FPGA (per-stage SNR vs the CPU reference is on by default; --max-seconds is the compile cap
# and part of the image signature, so keep it constant between runs)
python models/moonshine/moonshine_test.py --dev xdma0 --max-seconds 6 [--audio x.wav] [--no-snr] [--clean] [--verbose]
python models/moonshine/moonshine_test.py --dev xdma0 --max-seconds 6 --engines 8   # stem+encoder sharded
```

## Hardware obligations (from the 5-check gate)

| Check | Status |
| --- | --- |
| innermost ÷64 | hidden 416 → pad 448; intermediate 1664 ok; vocab 32768 ok |
| sub-64 split | head_dim 52 → pad 64; RoPE is partial (32 lanes, interleaved pairs) + 20 pass-through, needs its own cos/sin table |
| 4GB fit | 123 MB bf16 params; activations tiny; instruction stream dominated by 194 unrolled decode steps → PBI |
| URAM A/B | standard; cross-attn K/V are "weights" (B side) once computed |
| fixed shapes | cap audio at `max_audio_seconds`, decode at 194 steps, mask after EOS |

Decoder gated MLP: `x, gate = fc1(h).chunk(2)`; output `silu(gate) * x` (gate is the
second chunk). Attention scaling is `52**-0.5` even though HF pads head_dim to 56.
