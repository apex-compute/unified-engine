# Native 8-kHz DPDFNet2

This directory uses CEVA's official `dpdfnet2_8khz` model, pinned to SHA256
`6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d`.
Its native sample rate is 8 kHz, with a 160-sample Vorbis window, 80-sample
hop, 81 frequency bins and 37,860 recurrent-state values. The ONNX graph
has 492 operations. The existing 16-kHz port remains under `models/dpdfnet/`.

Status: CPU audio-to-audio execution is verified. The initial whole-graph FPGA
compiler and runner target RK AXI-256 with queue-CONFIG convolution. FPGA
accuracy and the 10-ms real-time deadline are **not yet verified**: Italy's
device changed to an incompatible AXI-512 build during implementation.
See [validation results](VALIDATION.md).

Install the model-specific dependencies from the repository root, using an
environment with the repository's matching `torch` and `torchaudio` packages:

```bash
python -m pip install -r models/dpdfnet8khz/requirements.txt
```

Run the native 8-kHz model on CPU:

```bash
python models/dpdfnet8khz/dpdfnet8khz_run_cpu.py --download \
  --input path/to/noisy.wav \
  --output models/dpdfnet8khz/dpdfnet8khz_bin/enhanced_cpu.wav
```

The first run downloads and verifies the official ONNX model in the ignored
`dpdfnet8khz_bin/` directory. Later runs reuse the verified file; `--model`
selects another local copy of the same pinned model. ONNX Runtime uses one
CPU thread, and the final `TEST_RESULT` JSON separates neural inference
time from audio preprocessing, reconstruction and session loading.

Input audio is downmixed to mono and resampled to 8 kHz only when necessary.
STFT and iSTFT run on the host. Output is a mono floating-point WAV at the
original input sample rate and exact sample count. A four-hop model delay
(320 samples, 40 ms at 8 kHz) is compensated after appending 480 samples of
silence to preserve the end of the recording. The default output has no
attenuation mixing, normalization or clipping. The optional
`--attn-limit-db` limits suppression by blending the aligned noisy spectrum.

Model source: [CEVA DPDFNet](https://github.com/ceva-ip/DPDFNet) and the
[official model download](https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx/dpdfnet2_8khz.onnx).
The model's metadata supplies all 81 ERB and 80 spectral normalization
initial values; the remaining recurrent state starts at zero.

Build a single deployment bin without accessing the FPGA:

```bash
python models/dpdfnet8khz/dpdfnet8khz_compile.py --download --force
```

On a compatible RK-256 queue-CONFIG build, run audio from that bin:

```bash
python models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --device rk --dev xdma0 \
  --input test_samples/p232_007.wav \
  --output models/dpdfnet8khz/dpdfnet8khz_bin/enhanced_fpga.wav
```

The resident program evaluates all 492 graph operations and commits recurrent
state, with one input upload, START/HALT sequence and output read per 80-sample
hop. Neural operations stay on the FPGA; audio framing and reconstruction run
on the host. The runner verifies the artifact checksum and compiled AXI width
before resetting or uploading. `--device rk` labels the target; it cannot
change the physical FPGA build. The current AXI-512 build is rejected.

An opt-in compiler optimization batches packed-state copies and reduces
convolution transpose work. Keep a separate artifact for comparison:

```bash
python models/dpdfnet8khz/dpdfnet8khz_compile.py --optimize --force \
  --output models/dpdfnet8khz/dpdfnet8khz_bin/dpdfnet2_8khz-optimized.bin
```

This reduces the initial program from 53,890 to 33,342 instructions. It remains
experimental until baseline and optimized outputs match on compatible hardware.
Instruction count alone does not establish latency.

The runner writes per-frame hardware and host timings beside its output as
`OUTPUT.metrics.json`; `--report` selects another path. Its final `TEST_RESULT`
records the metrics checksum, model/input/output hashes and detected FPGA
build. Mean, p95, p99, maximum and deadline misses are reported separately.
Frame times exclude startup, model upload, STFT/iSTFT and file I/O. Audio runs
also report total audio processing time excluding startup. Profiling with
`dpdfnet8khz_profile.py` replays operation prefixes and is not a latency test.

At 8 kHz, 80 samples still arrive every **10 ms**. The published 1.29 GMAC
figure is only 4.4% below the 16-kHz model's 1.35 GMAC, so changing sample rate
does not guarantee real time. Acceptance requires correct output, sustained
processing below the hop interval, and reporting tail latency and misses.
The model's separate 40-ms algorithmic delay is not a processing-time budget.

Prepare CPU references from one or more noisy recordings in a new directory:

```bash
python models/dpdfnet8khz/dpdfnet8khz_prepare_validation.py --download \
  --input test_samples/p232_007.wav \
  --output models/dpdfnet8khz/dpdfnet8khz_bin/validation
```

Repeat `--input` to add recordings. The preparer also creates silence and
speech/silence/speech cases with state retained throughout each case. On the
compatible FPGA, run all saved inputs from the same deployment bin:

```bash
for input in models/dpdfnet8khz/dpdfnet8khz_bin/validation/*_input.npy; do
  case_base="${input%_input.npy}"
  python models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
    --device rk --dev xdma0 --input "$input" --output "${case_base}_fpga.npy" \
    > "${case_base}_fpga.log" 2>&1 || break
done
python models/dpdfnet8khz/dpdfnet8khz_compare.py \
  --reference-manifest models/dpdfnet8khz/dpdfnet8khz_bin/validation/cpu_reference_manifest.json \
  --fpga-dir models/dpdfnet8khz/dpdfnet8khz_bin/validation \
  --output models/dpdfnet8khz/dpdfnet8khz_bin/validation/comparison.json
```

Comparison verifies file hashes, recomputes latency percentiles from every
measured frame, and requires all cases to pass a coarse CPU agreement check
and the measured deadline. It rejects timing with traces, overridden clocks,
mixed deployments or missing records. Default error limits (25% spectrum
relative L2 and 1e-6 absolute error for zero-energy silence) detect gross
execution failures; they are not speech quality acceptance criteria. A passing
`realtime_smoke_pass` covers only the measured corpus and neural frame loop.

Run host-only checks:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s models/dpdfnet8khz -p 'test_*.py'
```
