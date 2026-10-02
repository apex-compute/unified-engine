# FPGA Token2Wav attempt report

Date: 2026-10-01. Branch: `omni/talker-speech`, after FPGA Talker commit
`21120bed`.

## Result

Token2Wav is **not on the FPGA**. The default `--speak` path still runs the
eight-engine FPGA Talker, then loads the FP32 Token2Wav model on the host and
writes the waveform there. I did not change that path or claim a hardware-verified
vocoder. The immediately preceding high Omni hardware run ended with
`TimeoutError: prefill master is still busy after 546.1s`; per the standing
no-more-FPGA-runs-after-timeout instruction, no further FPGA job was launched.
The requested `make reboot_from_flash` could not run from `unified-engine`
because that Makefile has no such target. A board recovery has not been
verified in this task.

## Measured work and exact model structure

The local checkpoint contains **809 FP32 Token2Wav tensors, 1,713.0 MiB**:
1,272.6 MiB in `code2wav_dit_model` and 440.4 MiB in
`code2wav_bigvgan_model` (safetensors metadata, without loading tensor data).
The production Chelsie summary reports 150 codec tokens, 3.00 s of audio,
and 10.7 s CPU wall for Token2Wav, including model construction and weight
loading.

A CPU-only probe using this exact checkpoint, Chelsie conditioning, and 150
synthetic codec IDs measured **7.801 s DiT** and **1.327 s BigVGAN**. It produced
`[1, 80, 300]` mel and 72,000 waveform samples. Synthetic IDs exercise the
same shapes but are not a quality test; these numbers do not predict FPGA
performance.

The default sampler uses ten time points and a four-evaluation RK4 step for
each of nine intervals: **36 DiT forward evaluations**. Each evaluation has
22 transformer blocks at hidden width 1,024, 16 attention heads of width 64,
block-local noncausal masks, a two-way classifier-free-guidance batch,
time/AdaLayerNorm conditioning, and a speaker ECAPA encoder. The final
BigVGAN has six transposed-convolution upsample stages with rates
`[5, 3, 2, 2, 2, 2]` (240x total), three AMP residual blocks per stage,
six 1-D convolutions and six SnakeBeta activations per AMP block, plus input
and output convolutions. Its SnakeBeta includes sine and depthwise up/down
filters. The model's implementation runs all of this in FP32.

Sources: `qwen2.5_omni_7b_test.py` (`_run_fpga_speech`, Token2Wav call), the
installed Transformers `modeling_qwen2_5_omni.py` (`Qwen2_5OmniToken2WavDiTModel`,
`RungeKutta4ODESolver`, `Qwen2_5OmniToken2WavBigVGANModel`), and the local
checkpoint `config.json`.

## FPGA gap, not an impossibility proof

* Existing `matmat_mul_core`, normalization, elementwise, and attention paths
  cover substantial portions of DiT. They do **not** constitute a compiled
  Token2Wav pipeline: 36 repeated full-sequence evaluations, CFG batch,
  ECAPA temporal convolutions/pooling, timestep modulation, block masks,
  and RK4 state updates still need scheduling and validated scratch layout.
* There is no direct FP32 matmul mode in `user_dma_core.TYPE`; current tensor
  paths use BF16 and IF4/IF8 weights. Lowering this FP32 vocoder to BF16 or
  quantized weights is a numerical change that must be checked against mel
  and waveform references, not assumed safe. The existing FPGA Talker also
  has a known IF4 codec-token accuracy issue, so a vocoder-quality test must
  distinguish Talker error from Token2Wav error.
* BigVGAN is convolution-heavy, not merely a transformer tail. Kokoro's
  `kokoro_fpga.py` provides compositional shifted-matmul 1-D convolution,
  zero-insert transposed convolution, and bounded sine approximation that
  could be adapted. It does not yet implement this BigVGAN graph or prove
  its performance or waveform fidelity.
* The speech DRAM map currently reserves 544 MiB private weight space and
  184 MiB private tensor space per engine for Thinker plus Talker, with only
  about 200 MiB shared space per engine. Token2Wav's 1,713 MiB FP32 weights
  cannot simply be added to that live map. After codec generation, Thinker
  and Talker weights could be released and a **separate Token2Wav stage map**
  established. Evenly partitioned, the FP32 weights average about 214 MiB
  per engine; actual placement and convolution sharding require a new audit.

## Smallest credible implementation sequence

1. Save fixed *correct* codec IDs, Chelsie conditioning, reference mel, and
   FP32 mel/waveform checkpoints from a CPU reference. Do not use the
   current sampled FPGA Talker output as the only correctness oracle.
2. Port and validate one DiT forward evaluation first: speaker encoding,
   input embeddings, all 22 blocks, final mel projection. Compare layerwise
   SNR/absolute error with the FP32 reference. Then add two-way CFG and the
   36-evaluation RK4 loop, keeping mel/RK4 state on-device.
3. Reuse Kokoro's conv/transpose-conv construction for BigVGAN; implement
   its exact padding, dilations, residual aggregation, SnakeBeta, and
   depthwise filters. Compare mel-to-waveform output separately before
   connecting it to DiT.
4. Reclaim Thinker/Talker DRAM only after codec generation, load Token2Wav
   shards into an explicitly checked new map, and report CPU and hardware
   counters for DiT, BigVGAN, transfers, and end-to-end audio. Only after
   board recovery, run one short hardware case with a retained runtime log.

No FPGA Token2Wav source or unverified default-path switch was committed in
this attempt.
