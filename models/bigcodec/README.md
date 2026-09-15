# BigCodec

[BigCodec](https://github.com/Aria-K-Alethia/BigCodec) compresses speech into discrete tokens and reconstructs speech from those tokens.

| Property | Value |
| --- | --- |
| Native sample rate | 16,000 Hz |
| Token interval | 200 samples / 12.5 ms |
| Codebook | 8,192 entries; 13 bits per token |
| Nominal token bitrate | 1,040 bits/s |
| Parameters | 159,436,290, counting upstream weight normalization parameters |
| Execution | Complete utterance; centered convolutions and two residual LSTM stacks |

The corrected FPGA implementation runs the complete encoder, quantizer and decoder. On the [eight 20–24-second noisy files previously tested with DPDFNet](validation/20260915_accuracy/decoder/README.md), pooled waveform error against the official FP32 CPU model falls from **28.06% to 23.94% with BF16 recurrence**, and from **27.04% to 24.35% with IF8 recurrence**. BF16 improves on all eight files; IF8 improves on seven. The separate short diagnostic clip regresses slightly, as documented in the [accuracy report](validation/20260915_accuracy/README.md).

Processing RTF is **4.82 for BF16**, **4.38 for IF8 recurrence** and **2.32 for one-thread CPU FP32**. Each FPGA run uses one bin and one START/HALT per complete file. Processing RTF excludes startup; including measured runner startup gives **7.18** and **6.67**, respectively. Real time requires RTF ≤ 1. The remaining waveform error is material; this port does not match FP32 exactly.

The [previous 3.956-second benchmark](validation/20260914_optimized_italy/README.md) measured the optimization from RTF 15.32 to 4.84 with BF16 waveform samples and tokens bit-identical to the earlier FPGA implementation; IF8 recurrence reached 4.40. The [earlier benchmarks](validation/20260914_italy/README.md) and [initial optimization estimates](validation/20260914_optimization/README.md) retain their original measurements.

## Setup

From the repository root, in a Python environment with the FPGA driver's dependencies:

```bash
python -m pip install -r models/bigcodec/requirements.txt
python models/bigcodec/bigcodec_fetch.py
```

The downloader verifies the pinned checkpoint SHA256. See [upstream_manifest.json](upstream_manifest.json) and [NOTICE.md](NOTICE.md) for versions and licenses. Checkpoints and generated bins stay in the ignored `bigcodec_bin/` directory.

## Audio file to audio file on FPGA

Compile for the input file's length, then execute on the RK AXI256 board:

```bash
python models/bigcodec/bigcodec_compile.py \
  --input input.wav --conv-precision bf16 \
  --lstm-cell-precision compensated --lstm-tanh-precision compensated \
  --lstm-fused-gates --lstm-math-scope decoder \
  --center-quantizer-scores --compensated-codebook \
  --output models/bigcodec/bigcodec_bin/bigcodec-andromeda.bin

python models/bigcodec/bigcodec_run_from_bin.py \
  --input input.wav --output reconstructed_fpga.wav
```

Use `--force` to replace an existing compiled bin. `--conv-precision bf16` reuses overlapping input windows in SRAM for BF16 matrix multiplication; `--conv-precision if8` uses native IF8 convolutions and is the CLI default. BF16 was faster and had lower end-to-end waveform error in the baseline measurements. Compilation is offline and requires the checkpoint. Execution needs only the deployment bin and input audio. The bin contains both packed parameters and the captured instruction program.

The command selects the [corrected numerical implementation](validation/20260915_accuracy/README.md). Decoder LSTM updates retain a BF16 high and low cell value, tanh retains intermediate residuals, and recurrent projection, bias and sigmoid share one writeback. Centered codebook scores retain close differences; spare matrix lanes hold codebook weight residuals. These corrections execute on the FPGA and require no new bitstream. Omitting these numerical options retains the original arithmetic.

The prepared default bin on Italy uses this BF16 profile for the 63,400-sample padded `p232_007` input. [Deployment hashes and the preserved original bin](validation/20260915_accuracy/deployment.json) are recorded. Recompile for another padded input length.

Recurrent weights default to BF16. Add `--lstm-precision encoder-if8` to stream IF8 recurrent weights in the encoder, or `--lstm-precision if8` for both encoder and decoder. Input projections remain BF16. The earlier short-sample benchmark records RTF 4.62 for encoder-only IF8 and 4.40 for both stacks, before the numerical corrections.

The runner uploads the model image once, writes the padded input to DRAM, issues one START, waits for the terminal HALT, then reads one output bundle containing the waveform and tokens. All neural operations, including recurrent state updates and codebook selection, execute on the FPGA. Host processing reads/resamples the input and writes the output WAV and token NPZ.

The mono output preserves the source duration and sample rate; multichannel inputs are averaged before encoding. Each bin accepts a fixed padded native length and can be reused for inputs with that length. Matching upstream, padding always adds `200 - (samples % 200)` zeros, including a full extra hop for aligned inputs. The complete utterance resets recurrent state. Independent 10 ms calls are not equivalent to this model's inference.

For longer utterances, the compiler automatically expands the model/program arena, first to 1 GiB and then to 1,216 MiB if needed, while reserving 768 MiB for tensors within the RK-256 board's 2 GiB DRAM. The largest program layout reserves 64 MiB for padded input. This allows the full 20–24-second noisy test files to run in one execution. `--memory-layout large-program` selects that layout directly and avoids compilation retries. The runner checks the board's reported capacity before uploading; shorter files retain the original memory layout.

The command also writes `reconstructed_fpga.tokens.npz` and `reconstructed_fpga.metrics.json`. NPZ stores uint16 token IDs with audio metadata; it is not a packed 13-bit transport stream.

## CPU comparison

```bash
python models/bigcodec/bigcodec_run_cpu.py \
  --input input.wav --output reconstructed_cpu.wav \
  --tokens reconstructed_cpu.tokens.npz --threads 1 --cpu-core 7
```

To reconstruct a saved token file independently:

```bash
python models/bigcodec/bigcodec_run_cpu.py --mode decode \
  --tokens reconstructed_cpu.tokens.npz --output decoded_cpu.wav \
  --threads 1 --cpu-core 7
```

Compare the two reconstructions without fitting gain or timing:

```bash
python models/bigcodec/bigcodec_compare.py \
  --reference reconstructed_cpu.wav --actual reconstructed_fpga.wav \
  --reference-tokens reconstructed_cpu.tokens.npz \
  --actual-tokens reconstructed_fpga.tokens.npz --report comparison.json
```

RTF is processing time divided by input audio duration; a value below 1 means faster than real time. The FPGA report's `audio_rtf` includes preprocessing, input/output DMA, FPGA execution and output restoration/write. It excludes bin loading, model upload and compilation. Hardware cycle timing and upload time are recorded separately.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s models/bigcodec -p 'test_bigcodec_*.py'
```

Tests cover upstream source integrity, token/audio metadata, convolution padding and transpose phases, activation math, LSTM gates/state reset, quantizer selection, graph memory planning and the single START/HALT runtime protocol. Hardware results are reported separately from software emulation.

The FPGA Snake approximation folds its BF16 argument into a sine period and evaluates a degree-10 sine-squared polynomial, clamping argument magnitude at 32π. Quantized alias filters preserve symmetry and exact unit DC gain. Tanh uses an odd Padé [7/6] approximation with arguments clamped to ±4 and output to ±1, preserving quiet values that `2*sigmoid(2*x)-1` loses in BF16. Codebook comparisons use BF16 scores and choose the lowest token ID on ties. On build `0x40519e0a`, positive score gaps below 2⁻¹²⁷ become ties; zero signs can differ. These precision changes can alter tokens and reconstructed audio.
