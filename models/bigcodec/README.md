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

The CPU reference uses the pinned official FP32 checkpoint. The FPGA implementation runs the complete encoder, quantizer and decoder using either IF8 or BF16 convolution weights. It remains **experimental**: token choices and reconstructed waveforms differ from FP32, and both measured FPGA modes are slower than real time. See the [benchmarks and audio samples](validation/20260914_italy/README.md).

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
  --output models/bigcodec/bigcodec_bin/bigcodec-andromeda.bin

python models/bigcodec/bigcodec_run_from_bin.py \
  --input input.wav --output reconstructed_fpga.wav
```

Use `--force` to replace an existing compiled bin. `--conv-precision bf16` uses device im2col and BF16 matrix multiplication; `--conv-precision if8` uses native IF8 convolutions and is the CLI default. BF16 was faster and had lower end-to-end waveform error on the reported samples. Compilation is offline and requires the checkpoint. Execution needs only the deployment bin and input audio. The bin contains both packed parameters and the captured instruction program.

The runner uploads the model image once, writes the padded input to DRAM, issues one START, waits for the terminal HALT, then reads one output bundle containing the waveform and tokens. All neural operations, including recurrent state updates and codebook selection, execute on the FPGA. Host processing reads/resamples the input and writes the output WAV and token NPZ.

The mono output preserves the source duration and sample rate; multichannel inputs are averaged before encoding. Each bin accepts a fixed padded native length and can be reused for inputs with that length. Matching upstream, padding always adds `200 - (samples % 200)` zeros, including a full extra hop for aligned inputs. The complete utterance resets recurrent state. Independent 10 ms calls are not equivalent to this model's inference.

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

The FPGA Snake approximation folds its BF16 argument into a sine period and evaluates a degree-10 sine-squared polynomial, clamping argument magnitude at 32π. Quantized alias filters preserve symmetry and exact unit DC gain. Tanh uses an odd Padé [7/6] approximation with arguments clamped to ±4 and output to ±1, preserving quiet values that `2*sigmoid(2*x)-1` loses in BF16. Codebook comparisons use BF16 scores and choose the lowest token ID on ties; positive BF16 subnormal differences flush to zero on this board. These precision changes can alter tokens and reconstructed audio.
