# FLUX.2 [klein] 4B

Text-to-image on the Unified Engine: the full `black-forest-labs/FLUX.2-klein-4B`
pipeline (Qwen3-4B text encoder, 5 double + 20 single-stream diffusion
transformer blocks, 4 flow-matching Euler steps, AutoencoderKLFlux2 decoder)
with every matmul, norm, attention, activation and convolution on the FPGA.
The host only tokenizes, draws the seed noise, moves tensors between the
three phases and writes the PNG.

| File | Purpose |
|---|---|
| `flux2_klein_4b_test.py` | Build + run. Exports weights on first use, compiles the two programs, runs TE -> DiT -> VAE, scores against the host IF4 reference, prints `TEST_RESULT`. `--compile-only` needs no board. |
| `flux2_klein_4b_run_from_bin.py` | Run-only: one configuration per call from the prebuilt weight files and that configuration's compiled programs (never quantizes or compiles; no host reference unless `--reference`). |
| `flux2_klein_4b_server.py` | Minimal web UI (standard library only): prompt box, resolution picker over the compiled program sets, seed; runs `flux2_klein_4b_run_from_bin.py` one job at a time and shows the PNG. Selecting a configuration shows its weight images, program sizes and FLOPs per phase; a finished run shows every phase's on-device GFLOPS and share of the raw peak (128 FLOP/cycle/engine at the HW_INFO clock). |
| `flux2_klein_4b_vae.py` | Phase 3: planner + emitter of the VAE decoder as one sharded program (convs straight from the padded maps, 1x1 convs as matmuls, on-device GroupNorm statistics, SiLU, residual add, upsample, mid-block attention), plus its IF8 host oracle with per-op intermediates. |
| `flux2_klein_4b_vae_probe.py` | On-board bring-up of the VAE phase: compiles any prefix of the decoder graph and scores the last op against the host intermediate. |
| `flux2_klein_4b_weights.py` | Checkpoint -> `params_te.bin` (IF4 Qwen3), `params_dit.bin` (IF4 DiT), `vae_weights.pt` (IF8 convs); shape-only layout planner the compiler addresses from. |
| `flux2_klein_4b_ref.py` | Plain-torch reference of the whole pipeline (bit-identical to diffusers in fp32); IF4 / IF8 fake-quant helpers. |
| `flux2_klein_4b_compare_diffusers.py` | Reference vs the official diffusers pipeline (needs a GPU). |
| `flux2_klein_4b_quant_report.py` | BF16 vs IF4 quality (prompt-embedding SNR, latent SNR, image PSNR). |
| `test_flux2_klein_4b.py` | Host-only unit tests (geometry, RoPE permutation, packing, offline compile, diffusers parity). |
| `flux2_klein_4b_config.json` | Model geometry, quantization policy, DRAM layout, paths. |
| `flux2_klein_4b_bin/` | Generated: weight images, programs, PNGs, reports (git-ignored). |

## Usage (repo root)

```bash
# one-time: environment with the pinned stack + diffusers (reference / comparisons)
python3 -m pip install -r requirements.txt diffusers accelerate safetensors

# build everything and generate an image (first run downloads ~10 GB, quantizes ~3 min, compiles ~5 s)
python models/flux2_klein_4b/flux2_klein_4b_test.py --prompt "A photo of a red sports car on a mountain road at sunset"

# quicker CI geometry (256x256, 128 text tokens) and host-decoded VAE (standard FPGA image)
python models/flux2_klein_4b/flux2_klein_4b_test.py --height 256 --width 256 --max-seq-len 128 --vae host

# compile the programs of a configuration (no board needed, ~15 s each; cached under flux2_klein_4b_bin/programs/)
python models/flux2_klein_4b/flux2_klein_4b_test.py --compile-only --height 256 --width 256 --max-seq-len 64 --engines 8
python models/flux2_klein_4b/flux2_klein_4b_test.py --compile-only --height 512 --width 512 --max-seq-len 64 --engines 8

# run one configuration from the prebuilt files (no quantization, no compile): ~38 s for 256x256, ~2 min 10 s for 512x512 on 8 engines
python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --height 256 --width 256 --max-seq-len 64 --engines 8 --prompt "A photo of a red sports car on a mountain road at sunset" --out car_256.png
python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --height 512 --width 512 --max-seq-len 64 --engines 8 --prompt "A photo of a red sports car on a mountain road at sunset" --out car_512.png

# web UI on http://127.0.0.1:8080/ (pick a compiled resolution, type a prompt, see the image); --cpus pins the runs
python models/flux2_klein_4b/flux2_klein_4b_server.py --port 8080 --cpus 0,2-23

# host checks
python -m unittest discover -s models/flux2_klein_4b -p 'test_*.py'
python models/flux2_klein_4b/flux2_klein_4b_compare_diffusers.py --height 256 --width 256 --max-seq-len 128
python models/flux2_klein_4b/flux2_klein_4b_quant_report.py --height 512 --width 512 --max-seq-len 512
```

| Flag | Default | Notes |
|---|---|---|
| `--height/--width` | 512 | multiples of 16; image tokens = (H/16)*(W/16) must be a multiple of 64 |
| `--max-seq-len` | 512 | text tokens (padded, like the pipeline); multiple of 64. `auto` = the prompt's own length rounded up to 64 (a 25-token prompt -> 64), which removes a third of the DiT work at 512x512 |
| `--steps` | 4 | distilled schedule; the dt of every step is baked into the program |
| `--seed` | 0 | same CPU generator semantics as diffusers |
| `--vae fpga\|host` | fpga | `fpga` needs the CONV2D queue-config image |
| `--text-encoder fpga\|host` | fpga | `host` skips the TE phase (transformers on the host) |
| `--no-reference` | off | skip the IF4 host-reference scoring |
| `--ref-device` | cuda if available | device for the host reference |

Programs are keyed by a signature over geometry, engine count and the
compiler source; a different resolution, text length, step count, engine
count or code change recompiles (seconds). Weight files are geometry-independent.

## Prebuilt files

`flux2_klein_4b_weights.py` writes the geometry-independent weight files once
(`params_te.bin`, `params_dit.bin`, `embed_tokens.bin`, `vae_weights.pt`);
`flux2_klein_4b_test.py --compile-only` writes one program set per
configuration and engine count under `flux2_klein_4b_bin/programs/`
(`<HxWxTOKENSxSTEPS>_e<engines>.bin` plus its JSON: every engine's image and
its baked DRAM address). `flux2_klein_4b_run_from_bin.py` loads exactly those
files for the configuration given on its command line and DMAs them to the
board; it refuses a stale or missing program set instead of compiling.

## Multi-engine (`--engines N`, default `max` = HW_INFO core count)

Every op is emitted into each engine's own program with that engine's share:
matmuls are split over output columns (every engine reads the same IF4 weight
image; its column strip lands in the full output through the dynamic matmul's
row-stride override), norms / RoPE / eltwise / copies / scatters over rows,
attention over heads (3 of 24 DiT heads or 4 of 32 Qwen3 heads per engine,
per-engine scratch), and a four-phase flag barrier
(`MultiEngineScheduler.barrier_slice`, the pi05 prefix protocol) follows every
op. Per-engine fixed GPRs carry the engine's own row counts so the dynamic
cores run unchanged. Worker programs are extra images in `programs.bin` with
their own 144 MiB slots; the primary launches the workers, runs its stream and
joins them. With `--engines 1` the stream is byte-for-byte the single-engine
one. Ops that cannot split (the 4-row modulation `+1`, the 128-wide
`proj_out`) run on engine 0 while the others just take the barrier.

`flux2_klein_4b_probe.py --engines 8` runs every primitive, a whole double and
single block, the tail and a full step through the sharded streams against
torch; all pass at the single-engine SNRs.

## How it maps onto the engine

**Phase 1, text encoder.** Qwen3-4B, only the 27 layers whose outputs are
used, at M = `max_sequence_length` (the pipeline pads to that length). Per
layer: RMSNorm, IF4 q/k/v, per-head QK RMSNorm, rotate-half RoPE through the
GQA table, head-major permute, 32 `unified_attention_core` calls (one per
query head; the bias is causal plus a right-padding key mask built per prompt),
permute back, o_proj, residual, RMSNorm, SwiGLU (SiLU fused on the gate
matmul), down, residual. The raw residual after layers 9/18/27 is scattered
into the DiT's 7680-wide text buffer with a per-token PBI loop, so the prompt
embedding never leaves DRAM. The 778 MB token embedding stays on the host
(rows gathered per prompt), as in `qwen3_4b`.

**Phase 2, diffusion transformer.** One program, the 4 Euler steps unrolled
(the loop-free pattern pi05 settled on). Up front, the timestep sinusoids of
all steps (a host table) go through the timestep MLP and the three shared
modulation projections as M = 4 matmuls; `(1 + scale)` copies are formed with
an ADD_BROADCAST. Each step then:

* re-seeds the text rows from the context embedding and embeds the current
  latents (`x_embedder`) into the image rows of one joint `[text; image]`
  buffer;
* materialises the five gate vectors as row broadcasts (the only broadcast
  needed: `LN(x)*(1+scale)+shift` is exactly `layer_norm_core_dram` with
  gamma = the `(1+scale)` row and beta = the `shift` row);
* runs the double blocks (per-stream modulated LN, IF4 q/k/v, QK RMSNorm,
  joint RoPE, joint 24-head attention, per-stream out-proj, gated residual,
  modulated LN, SwiGLU FF, gated residual) and the single blocks (fused
  qkv+mlp projection split into five IF4 matmuls, parallel attention and
  SwiGLU, `to_out` split into two matmuls summed on device);
* applies `norm_out` / `proj_out` to the image rows and the Euler update
  `x += dt * v` with the step's baked `dt`; finally the BatchNorm
  de-normalisation `x * std + mean` of the latents.

**Phase 3, VAE decoder.** `AutoencoderKLFlux2` (32-ch latent, 2x2 patchify,
`post_quant_conv`, conv_in, mid resnet / single-head attention / resnet, four
levels of three resnets with nearest-2x upsample convs, GroupNorm/SiLU/conv_out)
as ONE program: 119 ops emitted into every engine's stream as the third phase
of the program set, with the scheduler's flag barriers between them. The host
DMAs the latent map and the constant image (unrepeated IF8 weight blocks,
matmul operands of the 1x1 convs, GroupNorm affine rows, identity / ones /
group-average matrices) in before the phase and reads the image out after it;
nothing moves between host and board while it runs. Every map lives in DRAM
channels-in-lanes with a one-pixel zero border kept by its producer. 3x3 convs
keep the layer driver's exact tile grid, each tile one strided DRAM->URAM read
of its window straight from the padded map, CONV2D, and strided writes of the
tile rows into the output map (tile ranges split over the engines; the walker's
per-pixel weight stream is expanded on-engine right before the conv). 1x1
convs with C, OC multiples of 64 are one quantized-B matmul per image row
(the map is already a row-major [pixels, C] matrix). GroupNorm accumulates
lane-wise pixel sums into slotted accumulators, transposes the flush blocks
with an identity matmul and sums them with a ones matmul on the bf20 ladder,
pools per group with a group-average matmul, does the same for the centred
squares, takes rsqrt by Newton iteration and applies `x*A_c + B_c`. SiLU is
the driver's max/exp/polynomial form on a device-built `[x, 0]` pair map; the
upsample is one read and four strided writes per input row; the attention runs
on compact (seq, C) maps the surrounding 1x1 matmuls write and read directly.

**RoPE.** FLUX rotates interleaved pairs `(2j, 2j+1)` on four 32-dim axes
(T, H, W, L; theta 2000). The engine core is HF rotate-half, so the exporter
permutes the Q/K projection rows (and q/k RMSNorm gammas) of every head to
even-first order and the per-token table is `[cos | cos | -sin | sin]` over the
pair frequencies; attention is invariant to the shared permutation.

**Precision.** IF4 (per-64-block min-MSE INT4/FP4, `quant_lib.quantize`) for
all Qwen3 and DiT block projections; BF16 for the embedders, timestep MLP,
shared modulation projections and the output head (`quantization.bf16_keys`);
IF8 for the VAE convolutions; BF16 activations throughout (BF19 internally).

**DRAM (U50, everything below the 4 GiB DMA limit).** Params arena 2.5 GiB at
0x0 holds the DiT image (2.35 GB) or the TE image (1.45 GB) — the two
time-share it and are DMA'd in at their phase (~2 s per run). Tensors from
0xA0000000 (~280 MB at 512x512, ~800 MB at 1024x1024), programs from
0xE8000000 (TE 10.7 MB + DiT 32.6 MB, geometry-independent thanks to the
dynamic-M cores).

## Validation ladder

| Check | Result |
|---|---|
| Reference vs diffusers, fp32 single forward (DiT, VAE) | bit-identical (`max|diff| = 0`) |
| Reference vs diffusers, bf16 pipeline 256x256 | prompt embeds exact; step-0 latents 55 dB; final latents 24 dB (bf16 kernel-order noise amplified by the front-loaded schedule); image PSNR 35 dB |
| Exported bins vs shape-only layout plan | match (asserted at load) |
| Offline compile 256x256/128 and 512x512/512 | TE 10.7 MB (0.7 / 2.9 TFLOP), DiT 32 MB (9.6 / 40.7 TFLOP) |
| BF16 vs IF4 host reference, 512x512/512 tokens, seed 0 | Qwen3 IF4 prompt embeds 19 dB; DiT step-0 velocity 37 dB; final latents 2 dB / image PSNR 16 dB (18 dB with BF16 text) — the images are visually equivalent (same scene, car, lighting); the pixel-level divergence is the 4-step schedule amplifying small per-step differences, exactly as BF16-vs-diffusers already drifts to 24 dB. Per-step numbers in `flux2_klein_4b_bin/quant_report_*.json` (`flux2_klein_4b_quant_report.py`) |
| On the board (Alveo U50 image: queue mode, 256-bit AXI, 8 GiB, 8 engines, 300 MHz; one engine used), 256x256 / 128 text tokens, seed 0 | **text encoder on FPGA**: 20 s, 35.1 GFLOPS, prompt embeds 40.8 dB vs the IF4 host oracle. **DiT on FPGA**: 1 step 68 s / 4 steps 270 s at 35.6 GFLOPS (93% of the 38.4 GFLOPS peak); step-0 velocity (one full 25-block forward) 31.7-33.1 dB, end-to-end latents 9.4 dB after 4 steps (schedule amplification, see below), image PSNR 21 dB and visually the same sample. **VAE decoder on FPGA**: 38 s, image PSNR 41.6 dB vs the IF8 host oracle on the 1-step latents. Primitive probes (`flux2_klein_4b_probe.py`): LN-mod 48 dB, RMSNorm 50 dB, RoPE 54-57 dB, SiLU 56 dB, eltwise 55 dB, copies/scatters exact. |

The FPGA output is scored against the host reference run with the *same*
quantized weights (`quantized_reference`), so the score isolates engine
arithmetic from quantization loss. Because the sampler amplifies any
per-step difference (see the BF16-vs-diffusers and BF16-vs-IF4 rows), the
meaningful accuracy number for bring-up is the **step-0 latent SNR** (`step_latents_snr_db[0]`, measured 42.9 dB on the board)
(a single DiT forward on identical inputs; the IF4 weights alone score about
37 dB there); end-to-end latent SNR and image PSNR mainly confirm the run
is finite and lands on the same image, not bit-level agreement.

## Cost and expectations

Measured on the U50, 4 steps, all phases on the FPGA, from the prebuilt files:

| Image clock | Configuration | Text encoder | DiT | VAE | Total on board |
|---|---|---:|---:|---:|---:|
| 367 MHz (andromeda run 41729405935) | 256x256, 64 tokens, 8 engines | 1.1 s | 23.4 s (341 GFLOPS) | 8.9 s | **35.8 s** |
| 367 MHz | 512x512, 64 tokens, 8 engines | 1.1 s | 82.5 s (342 GFLOPS) | 31.7 s | **117.7 s** |
| 333 MHz (andromeda run 41691268697, flashed), one-shot VAE | 256x256, 64 tokens, 8 engines | 1.2 s | 25.7 s (310 GFLOPS) | 5.0 s | **35.3 s** |
| 333 MHz (flashed), one-shot VAE | 512x512, 64 tokens, 8 engines | 1.2 s | 90.7 s (311 GFLOPS) | 17.0 s | **116.4 s** |
| 333 MHz (flashed), per-layer VAE programs | 256x256, 64 tokens, 8 engines | 1.2 s | 25.7 s | 7.3 s | 38.4 s |
| 333 MHz (flashed), per-layer VAE programs | 512x512, 64 tokens, 8 engines | 1.2 s | 90.7 s | 27.8 s | 122.2 s |
| 333 MHz (flashed), host-orchestrated VAE | 256x256, 64 tokens, 8 engines | 1.2 s | 25.7 s | 9.2 s | 38.5 s |
| 333 MHz (flashed), host-orchestrated VAE | 512x512, 64 tokens, 8 engines | 1.2 s | 90.7 s | 33.1 s | 127.4 s |

The one-shot VAE decoder scores 52.2 dB (256x256) / 55.5 dB (512x512)
image PSNR against the IF8 host oracle on the same latents (`vae_psnr_db`),
runs at 131 / 150 GFLOPS, and a run captures no instructions and moves
nothing between host and board during any phase (the 367 MHz rows above were
taken with the host-orchestrated VAE). Host-side, torch's intra-op pool defaults
to one thread per CPU for the per-layer packing; cap it with
`OMP_NUM_THREADS` if the host should stay quieter.

Earlier measurements on the 300 MHz image (same accuracy):

| Configuration | Text encoder | DiT | VAE | Total on board | vs IF4 oracle |
|---|---:|---:|---:|---:|---|
| 256x256, 64 tokens, 8 engines, sharded VAE | 1.3 s | 28.6 s | 9.7 s | **~40 s** | step-0 latents 45.7 dB, image PSNR 28 dB |
| 512x512, 64 tokens, 8 engines, sharded VAE | 1.3 s | 100.8 s | 34.8 s | ~2 min 20 s | step-0 latents 46.9 dB, image PSNR 21 dB |
| 512x512, 512 tokens, 8 engines | 10.5 s | 145 s | 151 s (engine 0 only) | ~5 min | step-0 latents 46.5 dB, image PSNR 26 dB |
| 256x256, 128 tokens, 8 engines | 2.6 s | 34.4 s | 38 s (engine 0 only) | ~1.3 min | step-0 latents 41.5 dB, image PSNR 21.5 dB |
| 256x256, 128 tokens, 1 engine | 20.1 s | 270 s | 38 s | ~5.5 min | step-0 latents 42.9 dB, image PSNR 21 dB |

The transformer phases sustain 91% of the 8-engine peak (280 of 307 GFLOPS
at 300 MHz, 341 of 375 GFLOPS at 367 MHz), so the 4-step DiT's floor at
512x512 is 75 s with 64 text tokens on the 367 MHz image; accuracy is
unchanged by sharding.
`--max-seq-len auto` pads the prompt only to the next multiple of 64 instead
of 512, the single biggest lever left at 512x512. The VAE decoder is one
program with no host traffic while it runs.

## Known limitations / follow-ups

* Bring-up order that worked: `--steps 1 --text-encoder host --vae host` (one DiT forward, scored as `velocity0_snr_db`), then the text encoder, then `--vae fpga`. `flux2_klein_4b_probe.py` checks every primitive in isolation on the model's own buffers.
* The VAE phase needs the conv bitstream; `--vae host` is the fallback on the standard image.
* The DiT -> VAE hop (unpack / unpatchify of the latents) is still a host step between the two phases; an on-device scatter would make the three phases one launch.
* `flux2_klein_4b_vae_probe.py --sweep ...` scores any prefix of the decoder graph on the board against the host intermediates; SNRs fall along the chain as bf16 arithmetic (GroupNorm re-scales the input's error by |mean|/sigma, SiLU lowers signal power), the end-to-end check is the image PSNR against the IF8 oracle.
* Weights are re-DMA'd every run (phased layout) rather than resident on an 8 GiB board.
* GELU is not used by this model (SwiGLU everywhere), so the engine's sigmoid-approximation GELU is not a factor.
