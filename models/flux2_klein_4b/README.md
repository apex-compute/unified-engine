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
| `flux2_klein_4b_pack.py` | Packs weights, tokenizer, VAE and the compiled programs of one configuration into one deployable `flux2_klein_4b_<config>_e<engines>.bin`. |
| `flux2_klein_4b_run_from_bin.py` | Runs from that single bin: no network, no HF checkpoint, no other file. |
| `flux2_klein_4b_vae.py` | Phase 3: the VAE decoder through the conv / GroupNorm / SiLU / upsample layer drivers, plus its IF8 host oracle. |
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

# one deployable file per configuration (~4.7 GB each: weights + tokenizer + that configuration's programs for 8 engines)
python models/flux2_klein_4b/flux2_klein_4b_pack.py --config 256x256x64x4 --engines 8
python models/flux2_klein_4b/flux2_klein_4b_pack.py --config 512x512x64x4 --engines 8

# run from a file alone (no network, no HF checkpoint): ~42 s for 256x256, ~2 min 20 s for 512x512 on 8 engines
python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --bin models/flux2_klein_4b/flux2_klein_4b_bin/flux2_klein_4b_256x256x64x4_e8.bin --prompt "A photo of a red sports car on a mountain road at sunset" --out car_256.png
python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --bin models/flux2_klein_4b/flux2_klein_4b_bin/flux2_klein_4b_512x512x64x4_e8.bin --prompt "A photo of a red sports car on a mountain road at sunset" --out car_512.png

# compile without a board (writes flux2_klein_4b_bin/programs.{bin,json})
python models/flux2_klein_4b/flux2_klein_4b_test.py --compile-only

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
count or code change recompiles (seconds). Weight bins are geometry-independent.

## Single deployable bin

`flux2_klein_4b_pack.py` writes `flux2_klein_4b_<config>_e<engines>.bin`: a header (JSON: model
config, the packed configurations, engine count, layout manifests, program
metadata) followed by 4 KiB-aligned sections -- the tokenizer files, the Qwen3
token embedding, the IF4 text-encoder and DiT weight images, the IF8 VAE
weights, and the compiled programs (every engine's images, ~90 MB) of one
configuration (`--config HxWxTOKENSxSTEPS`); one bin per configuration.
`flux2_klein_4b_run_from_bin.py --bin FILE` DMAs each section to its baked
address and runs; `--list` shows what a bin was built for.
On the board the run is still three host-sequenced phases: the text-encoder
program, the DiT program, then the VAE decoder's per-layer programs.

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
through `run_conv2d_layer` (IF8, per-(oc, tap, 64-ch) scales, bias),
`run_group_norm`, `run_silu_layer`, `run_nn_upsample_2x`,
`run_eltwise_add_layer` and one `unified_attention_core` program for the
4096-token mid attention. Host-orchestrated between layers, arenas reset per
layer (U-Net backend pattern).

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

Measured on the U50, 4 steps, all phases on the FPGA, from the packed bins:

| Image clock | Configuration | Text encoder | DiT | VAE | Total on board |
|---|---|---:|---:|---:|---:|
| 367 MHz (andromeda run 41729405935) | 256x256, 64 tokens, 8 engines | 1.1 s | 23.4 s (341 GFLOPS) | 8.9 s | **35.8 s** |
| 367 MHz | 512x512, 64 tokens, 8 engines | 1.1 s | 82.5 s (342 GFLOPS) | 31.7 s | **117.7 s** |
| 333 MHz (andromeda run 41691268697, flashed) | 256x256, 64 tokens, 8 engines | 1.2 s | 25.7 s (310 GFLOPS) | 9.2 s | 38.5 s |
| 333 MHz (flashed) | 512x512, 64 tokens, 8 engines | 1.2 s | 90.7 s (311 GFLOPS) | 33.1 s | 127.4 s |

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
of 512, the single biggest lever left at 512x512. The VAE decoder's
convolutions are spread over all engines (`conv2d_layer_sharded`: every
output-channel chunk and tile range split evenly, one program per engine per
layer); GroupNorm, SiLU, the upsample copies and the residual adds still run
on engine 0 and are now most of the VAE time.

## Known limitations / follow-ups

* Bring-up order that worked: `--steps 1 --text-encoder host --vae host` (one DiT forward, scored as `velocity0_snr_db`), then the text encoder, then `--vae fpga`. `flux2_klein_4b_probe.py` checks every primitive in isolation on the model's own buffers.
* VAE phase is host-orchestrated (one program per layer) and needs the conv bitstream; `--vae host` is the fallback on the standard image.
* The VAE is still host-orchestrated (one program per engine per layer); its non-conv ops run on engine 0 only. Folding it into a whole-graph program would remove the host round trips.
* Weights are re-DMA'd every run (phased layout) rather than resident on an 8 GiB board.
* GELU is not used by this model (SwiGLU everywhere), so the engine's sigmoid-approximation GELU is not a factor.
