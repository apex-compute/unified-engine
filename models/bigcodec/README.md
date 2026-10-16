# BigCodec

Run commands from `pcie_utils/`. [BigCodec](https://github.com/Aria-K-Alethia/BigCodec)
compresses speech into discrete tokens and reconstructs it. It does not suppress
background noise; use [DPDFNet](../dpdfnet/README.md) for speech enhancement.

| Property | Value |
| --- | --- |
| Native sample rate | 16 kHz |
| Token interval | 200 samples / 12.5 ms |
| Codebook | 8,192 entries; 13 bits per token |
| Nominal bitrate | 1,040 bits/s |
| Execution | Complete utterance, including encoder, quantizer and decoder |

## Compile and run

```bash
python3 -m pip install -r requirements.txt
python3 models/bigcodec/bigcodec_fetch.py
python3 models/bigcodec/bigcodec_compile.py \
  --input models/test_samples/apex.wav --conv-precision bf16 \
  --lstm-cell-precision compensated --lstm-tanh-precision compensated \
  --lstm-fused-gates --lstm-math-scope decoder \
  --center-quantizer-scores --compensated-codebook \
  --output models/bigcodec/bigcodec_bin/bigcodec-andromeda.bin
python3 models/bigcodec/bigcodec_run_from_bin.py \
  --input models/test_samples/apex.wav --output /tmp/bigcodec_fpga.wav \
  --device rk --dev xdma0
```

The downloader verifies the pinned checkpoint SHA256. [upstream_manifest.json](upstream_manifest.json)
and [NOTICE.md](NOTICE.md) record source revisions and licenses. Checkpoints,
deployment bins and output media remain local artifacts in `bigcodec_bin/`
or the selected output directory. Compilation is offline; add `--force` to
replace an existing bin. The direct runner accepts `--bin` and needs only the
compiled artifact and input audio.

The example uses BF16 dense weights and compensated decoder arithmetic.
LSTM cells and sigmoid gates retain high/low values, tanh retains intermediate
residuals, and codebook scores retain close differences. These operations stay
on the FPGA. `--conv-precision if8` selects native IF8 convolutions and is the
compiler default. Recurrent weights default to BF16; `--lstm-precision encoder-if8`
or `if8` changes the encoder or both stacks respectively. Input projections
remain BF16.

Each bin accepts one padded native input length. Recompile when that length
changes. Upstream padding adds `200 - (samples % 200)` zeros, including one
extra hop for aligned input. The runner uploads the model once, writes input to
DRAM, issues one START, waits for HALT and reads a waveform/token bundle. All
neural operations run on the FPGA. Audio reading, resampling and writing run
on the host; independent 10 ms calls are not equivalent to this codec.

The compiler expands program memory when needed within RK-256's 2 GiB DRAM;
`--memory-layout large-program` selects the largest layout directly. The runner
checks board capacity before upload. A compatible RK AXI-256 CONV/MAXPOOL
queue-CONFIG image is required. `--cpu-core` selects an available host core
(the runner defaults to core 6).

Output preserves the original duration and sample rate; multichannel input is
averaged to mono. Adjacent `.tokens.npz` and `.metrics.json` files contain uint16
token IDs, audio metadata and timing. Tokens are not packed into a 13-bit stream.

## CPU and quality comparison

```bash
python3 models/bigcodec/bigcodec_run_cpu.py \
  --input models/test_samples/apex.wav --output /tmp/bigcodec_cpu.wav \
  --tokens /tmp/bigcodec_cpu.tokens.npz --threads 1
python3 models/bigcodec/bigcodec_compare.py \
  --reference /tmp/bigcodec_cpu.wav --actual /tmp/bigcodec_fpga.wav \
  --reference-tokens /tmp/bigcodec_cpu.tokens.npz \
  --actual-tokens /tmp/bigcodec_fpga.tokens.npz --report /tmp/bigcodec-comparison.json

# Isolate decoder error by decoding the FPGA's tokens on CPU.
python3 models/bigcodec/bigcodec_run_cpu.py --mode decode \
  --tokens /tmp/bigcodec_fpga.tokens.npz --output /tmp/bigcodec_same_tokens_cpu.wav \
  --threads 1
```

Waveform agreement with the official CPU model includes encoder/token and decoder
error. Comparing against CPU decoding of identical FPGA tokens isolates decoder
error. Neither measures noise suppression. For perceptual and spectral metrics:

```bash
python3 -m pip install -r models/bigcodec/requirements-quality.txt
python3 models/bigcodec/bigcodec_audio_quality.py \
  --reference /tmp/bigcodec_cpu.wav --actual /tmp/bigcodec_fpga.wav \
  --report /tmp/bigcodec-quality.json
```

`audio_rtf` is audio processing time divided by input duration, including
preprocessing, DMA, FPGA execution and output writing. It excludes bin loading,
model upload and compilation; those costs are reported separately. RTF ≤ 1
means average real-time throughput.

```bash
python3 -m unittest discover -s models/bigcodec -p 'test_bigcodec_*.py'
```

Host tests cover source integrity, audio/token metadata, convolution layout,
activation and recurrent arithmetic, quantizer selection, memory planning and
the single START/HALT protocol. Hardware measurements require a separate run.
