# DPDFNet2 on Andromeda

This directory contains the 16-kHz `dpdfnet2` port from CEVA's Apache-2.0
DPDFNet repository. The source of truth is the official streaming ONNX model,
pinned by SHA256. It consumes one complex spectrum frame represented as real
FP32 `[1, 1, 161, 2]` plus a flat 45,424-element recurrent state, and returns
one enhanced frame plus the next state. STFT and iSTFT are intentionally host
operations and are not present in the ONNX graph.

Enhance a WAV directly using an existing bin on RK-256, from the repository root:

```bash
python models/dpdfnet/dpdfnet_run_from_bin.py --device rk --dev xdma0 \
  --input test_samples/apex.wav \
  --output models/dpdfnet/dpdfnet_bin/apex_enhanced_fpga.wav
```

The runner loads `dpdfnet_bin/dpdfnet2-andromeda.bin` by default; use `--bin`
to select another artifact. WAV and FLAC inputs are downmixed to mono and
resampled to 16 kHz for inference. Host STFT/iSTFT converts between audio and
the model's spectrum frames. Output is a mono, floating-point WAV at the
original sample rate and exact original sample count. No attenuation mixing,
normalization or clipping is applied. The four-frame model delay is compensated,
with enough trailing silence to recover the end of the recording. The CPU
reference uses the same audio framing and reconstruction.

The file runner also accepts the existing `.npy` spectrum format. Each mode
uploads the model once and uses one START/HALT per spectrum frame, including
frames needed to flush audio at the end. No bin rebuild is needed to use WAV I/O.

The deployment contract is hardware-only for the complete ONNX neural graph:
the compiler may use ONNX/PyTorch while building the artifact, but the eventual
`run_from_bin` path may not invoke ONNX Runtime or PyTorch neural operators, and performs no
host-side intermediate tensor transfers beyond boundary packing. One spectrum frame is uploaded,
one resident FPGA program is kicked, recurrent state remains in device DRAM,
and the enhanced frame is read after the program's sole HALT. Audio STFT/iSTFT
is outside the supplied ONNX ABI; a strict zero-CPU application must provide
and consume spectrum frames outside this runner.

Current status is **end-to-end, stateful, single-bin hardware inference** for
the complete 472-node ONNX graph. The deployment image contains 473 operation
ranges: one per ONNX node plus the final recurrent-state commit. The runtime
uses no CPU neural-network operators and performs no intermediate host tensor
transfers. The model image is uploaded once; each spectrum frame uses one input
write, one program kick, one HALT and one output read.

The checked-in source supports AXI-256 and AXI-512 devices. The generated ONNX
and `dpdfnet2-andromeda.bin` files remain local build products under the ignored
`dpdfnet_bin/` directory.

The current artifact format is `andromeda.dpdfnet2.streaming-v2`. Rebuild older
bins with `python models/dpdfnet/dpdfnet_compile.py --force`. Version 2 fixes
the square-root lowering for zero spectrum energy: `sqrt(0)` now produces
zero instead of `0 * infinity`, which previously poisoned recurrent state.
It also aligns packed-state transfers so filter coefficients cannot overwrite
the neighboring spectrum buffer. Both corrections run entirely on the FPGA.
The runtime also rejects nonfinite
logical outputs and stops the backend; a failed backend must be recreated to
reload clean recurrent state.

Exercise the production square-root lowering on zero, mixed-zero and positive
BF16 energies (including the smallest normal value) on RK-256:

```bash
python models/dpdfnet/dpdfnet_sqrt_smoke.py --device rk --dev xdma0
python models/dpdfnet/dpdfnet_state_copy_smoke.py --device rk --dev xdma0
```

The Italy RK-256 regression passed 1,025 frames across the original clip,
silence, and audio/silence transitions with no nonfinite outputs. Pure
silence produces exactly zero. The optimized compiler reduces FPGA time
from 23.98 to 13.45 ms/frame and host time from 25.06 to 14.39 ms/frame,
with bit-identical output to the corrected baseline. It batches GRU input
projections, retains recurrent work in SRAM, combines safe whole-row DMA
copies and skips unused transpose calculations. Rebuild existing v2 bins
with `--force` to apply these optimizations. Full-clip error remains 5.24%
relative L2 against FP32 CPU; see [BENCHMARK.md](BENCHMARK.md) for measurements,
remaining accuracy limits and reproduction commands.

Recorded environmental-noise tests now cover bus, café, living-room, office
and public-square conditions, with clean references and saved listening
samples. Mean FPGA PESQ improves from 2.063 to 2.736 on the fixed 20-clip
subset, but one speech-onset accuracy failure remains. See
[NOISY_TESTS.md](NOISY_TESTS.md) for all quality scores, regressions and commands
to download and run the cases.

Install the model-specific audio and inspection/reference dependencies:

```bash
python -m pip install -r models/dpdfnet/requirements.txt
```

Audio I/O uses `soundfile` and the repository's existing `torch`/`torchaudio`
dependencies. Running a bin does not invoke ONNX Runtime; ONNX packages are
used for compilation, inspection and the CPU reference.

Download, verify and inspect the official graph:

```bash
python models/dpdfnet/dpdfnet_prepare.py --download \
  --audit-json perf_logs/dpdfnet/dpdfnet2_operator_audit.json
```

The model is written below the ignored `models/dpdfnet/dpdfnet_bin/` directory
and must not be committed. Run the CPU streaming reference on a WAV. Input is
converted to 16 kHz for the model and converted back to its original sample
rate for output:

```bash
python models/dpdfnet/dpdfnet_run_cpu.py --download \
  --input path/to/noisy_16khz.wav \
  --output models/dpdfnet/dpdfnet_bin/enhanced.wav \
  --attn-limit-db 12
```

Compile the pinned ONNX graph into one stateful Andromeda image:

```bash
python models/dpdfnet/dpdfnet_compile.py --force
```

Run one or more precomputed spectrum frames entirely through the FPGA graph:

```bash
python models/dpdfnet/dpdfnet_run_from_bin.py \
  --input /tmp/dpdfnet_apex_16.npy \
  --output /tmp/dpdfnet_apex_16_hw.npy \
  --device rk --dev xdma0
```

Export the final frame's chronological 8192-event trace tail:

```bash
python models/dpdfnet/dpdfnet_run_from_bin.py \
  --input /tmp/dpdfnet_apex_16.npy \
  --output /tmp/dpdfnet_apex_16_hw.npy \
  --device rk --dev xdma0 \
  --trace-tail perf_logs/dpdfnet
```

Profile every graph operation using the compiled bin, including operations
outside the circular trace tail:

```bash
python models/dpdfnet/dpdfnet_profile.py --device rk --dev xdma0 \
  --input /tmp/dpdfnet_apex_16.npy \
  --output perf_logs/dpdfnet/profile.json
```

This diagnostic replays program prefixes with the same input and initial
state, then restores the resident program and initial state. Use the normal
runner for streaming performance. Validate optimized GRUs against the
original BF16 lowering and cropped transposes on RK-256:

```bash
python models/dpdfnet/dpdfnet_gru_smoke.py --device rk --dev xdma0
python models/dpdfnet/dpdfnet_transpose_smoke.py --device rk --dev xdma0
```

Run dependency-free helper tests from the repository root:

```bash
python -m unittest discover -s models/dpdfnet -p 'test_*.py'
```

Verify the first supported dense ONNX Conv against a quantized CPU reference
on live hardware. The script exposes that node's real activation from an
in-memory ONNX graph, automatically selects gather-IF8, and exercises its
height-zero/width-one padding through CONV2D:

```bash
python models/dpdfnet/dpdfnet_conv_smoke.py \
  --device rk --dev xdma0
```

Add `--all` to validate all 15 dense Conv nodes using their actual one-frame
ONNX intermediates. This is a layer-by-layer hardware conformance suite, not
a whole-graph performance benchmark.

The 13 depthwise Conv nodes have a separate hardware-only tap-wise emitter and
conformance test. ONNX Runtime supplies only the oracle inputs/outputs for this
test; it is not part of the eventual deployment runtime:

```bash
python models/dpdfnet/dpdfnet_depthwise_smoke.py \
  --device rk --dev xdma0
```

Validate all 22 Gemm and 14 MatMul nodes, including the batched projection
weights used by the squeezed linear layers:

```bash
python models/dpdfnet/dpdfnet_linear_smoke.py \
  --device rk --dev xdma0
```

Validate all eight last-axis LayerNormalization nodes. The test reports the
small numerical effect of the current hardware primitive omitting ONNX's
`epsilon=1e-5` separately from its hardware-vs-BF16 error:

```bash
python models/dpdfnet/dpdfnet_layernorm_smoke.py \
  --device rk --dev xdma0
```

Validate all 20 Relu, 19 Sigmoid, and 10 Tanh nodes using the accelerator's
LALU activation paths (Tanh is lowered as `2*sigmoid(2*x)-1`):

```bash
python models/dpdfnet/dpdfnet_activation_smoke.py \
  --device rk --dev xdma0
```

Validate all Add/Mul/Sub nodes. ONNX broadcasts are expanded when the artifact
is prepared, so runtime execution remains a plain FPGA elementwise operation:

```bash
python models/dpdfnet/dpdfnet_eltwise_smoke.py \
  --device rk --dev xdma0
```

The commands above select RK-256. AXI-512 hardware can be selected with
`--device bittware_512`; the reported current benchmarks are from Italy's
RK-256 device.
