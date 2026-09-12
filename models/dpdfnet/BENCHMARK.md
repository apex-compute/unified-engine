# DPDFNet2 validation on Italy RK-256

Measured on 2026-09-13 with the optimized `streaming-v2` artifact. Software
lowering changes reduce FPGA frame latency from **23.981 to 13.447 ms**
(43.93%, 1.78x throughput) with bit-identical output. Host frame latency falls
from **25.064 to 14.392 ms** (42.58%). The FPGA NaN fix remains intact,
including silence and transitions between audio and silence. The full-clip
numerical difference from the FP32 CPU reference remains 5.24% relative L2.
Objective speech-quality tests with recorded environmental noise are now
reported in [NOISY_TESTS.md](NOISY_TESTS.md), including regressions and a
reproducible speech-onset accuracy problem. Human listening quality has not
been scored.

## Software optimizations

- Batch GRU input projections in tiles of up to 16 rows. Keep recurrent
  weights, hidden state and gate intermediates in SRAM for each direction,
  preserving every original BF16 arithmetic writeback.
- Copy complete padded tensor rows with contiguous or strided DMA when
  source padding is proven zero. Partial rows and unsafe padding retain the
  element-copy path; neighboring packed state remains protected.
- Compute only the rows and columns consumed by complex packing/unpacking
  transposes, while explicitly zeroing output padding.

The FPGA image and clock are unchanged. The neural graph still executes
entirely on the FPGA, with one input upload and output read per frame.

| Compiled image | Corrected baseline | Optimized |
|---|---:|---:|
| Instructions, including one HALT | 80,784 | 40,264 |
| DMA descriptors | 48,960 | 19,154 |
| Uploaded model bytes | 19,160,448 | 17,863,808 |
| ONNX nodes / operation ranges | 472 / 473 | 472 / 473 |

The new prefix-replay profiler covers all 473 operations despite the
8,192-entry circular trace limit. Diagnostic timing puts each of the two
48-step GRUs at about 0.352 ms, down from 1.77 ms. The largest remaining
individual operations are coefficient reshapes (0.543 and 0.443 ms) and two
convolutions (0.426 and 0.423 ms). These initial-state diagnostic replays are
separate from the streaming timings below.

## What was fixed

The first invalid intermediate was the spectrum-energy square root at ONNX
node 19. CPU returned zero for zero energy; the FPGA lowering evaluated
`0 * rsqrt(0)`, producing NaN. The compiler now bounds only the reciprocal's
input by the smallest normal BF16 value, preserving the original zero for
the final multiplication. This correction executes on the FPGA.

Debugging silence also exposed four unaligned DMA transfers in the packed
recurrent state. Filter coefficients beginning at element 40,934 overwrote
the preceding six BF16 values: the last three complex spectrum bins. The
compiler now uses aligned copies and lane selection to preserve both sides
of these boundaries. State layout and streaming ABI remain unchanged.

Old v1 bins must be rebuilt. The runtime rejects nonfinite logical output
and stops that backend; recreating it reloads the initial recurrent state.
Outputs are not replaced with zeros or repaired on the CPU.

## Hardware and CPU checks

Each sequence below started with the model's initial state. Both backends
processed the same spectrum frames; all 1,025 FPGA and CPU outputs were finite.
Every optimized FPGA output was bit-identical to the corrected baseline,
including the sign of zero. Three additional full-clip runs of each artifact
also matched the baseline bit for bit.

| Sequence | Frames | FPGA nonfinite values | Relative L2 versus CPU |
|---|---:|---:|---:|
| Complete `test_samples/apex.wav` | 513 | 0 | 0.052356 |
| Silence from startup | 128 | 0 | Both outputs exactly zero |
| 128 audio + 128 silent + 128 audio frames | 384 | 0 | 0.041707 |

The original clip previously produced 16,546 NaNs across its final 53 frames,
starting at frame 461 (1-based). After only the square-root fix, all output
was finite but relative L2 was 0.557832. Fixing state alignment reduced this
to 0.052356, with RMSE 0.056119. Relative L2 measures spectrum error, not
perceptual quality, and the remaining difference is not assumed to be solely
quantization error.

The square-root hardware smoke passed zero, mixed-zero and normal BF16
energies from `2^-126` through `1e30`; maximum relative error on positive
values was 0.2553%. The state-copy hardware smoke matched all 51,008 BF16
values exactly, including neighboring state and padding. The optimized GRU
hardware smoke matched the original lowering on 8-, 17- and 48-step
bidirectional sequences with nonzero initial state (9,728 BF16 values).
Eleven cropped-transpose cases checked 45,056 BF16 values, including zero
padding and untouched neighbors. An offline audit found all 19,154 optimized
DMA descriptors had 64-byte-aligned bases and stride jumps.
After the corrected baseline's transition run, all 512 materialized graph tensors and state
layouts were finite; two fused, unused transpose buffers were excluded.
Host regressions also passed: 34 DPDFNet tests, 8 initialization tests and
53 YOLO/trace tests (95 total), including row-copy padding, GRU recurrence
and profiler restoration checks.

## Timing

FPGA measurements are the median of three run averages per artifact, with
baseline and optimized runs alternating on the same device. Each run starts
with fresh model state and processes all 513 original-clip frames, including
recurrent-state updates. One frame represents a 10 ms hop.

| Measurement | Baseline ms/frame | Optimized ms/frame | Optimized real-time factor |
|---|---:|---:|---:|
| RK-256 FPGA execution counter | 23.981 | 13.447 | 1.345 |
| RK-256 including host frame handling and transfers | 25.064 | 14.392 | 1.439 |

Optimized FPGA run averages ranged from 13.44684 to 13.44744 ms/frame; host
averages ranged from 14.37987 to 14.44161 ms/frame. The earlier one-thread
ONNX Runtime FP32 CPU measurement on the same input was 0.644 ms/frame
(RTF 0.064); CPU was not retimed for this optimization.

The FPGA remains slower than real time. Median one-time optimized model upload
took 6.771 ms for 17,863,808 bytes, excluded from frame timings. Loading the bin,
file I/O and STFT/iSTFT are also excluded. The FPGA runner uploads the model
once, then uses one input write, program launch, HALT and output read per
frame, with no CPU neural operators or intermediate host transfers.

For context, Figure 5 of the [DPDFNet paper, version 3](https://arxiv.org/pdf/2512.16420v3#page=10)
reports DPDFNet-2 RTF 0.546 on NeuPro-Nano 32 and 0.442 on NeuPro-Nano 64.
These correspond to 5.46 and 4.42 ms per 10 ms hop. The paper uses INT8
weights and INT16 activations with quantization-aware training; this port
uses BF16/IF8-INT. These are different implementations and measurement
conditions, so the figures are contextual references, not controlled speedups.

## Reproduce on Italy

For audio-file input and output, the existing bin runner also accepts WAV:

```bash
source /home/hunlu/my_torch_env/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python models/dpdfnet/dpdfnet_run_from_bin.py \
  --device rk --dev xdma0 --input test_samples/apex.wav \
  --output models/dpdfnet/dpdfnet_bin/apex_enhanced_fpga.wav
```

This path was checked on the same RK-256 and optimized bin. The output is a
finite mono WAV with the original 48 kHz rate and all 244,880 samples
(5.101667 s). It executes 517 spectrum frames, including four additional
frames compared with the historical 513-frame input above: the audio wrapper
now pads enough silence to flush the model's 40 ms delay without cutting off
the last 20–30 ms. A 321-sample silent WAV remains exactly zero. The CPU audio
reference shares these framing and reconstruction fixes. The reconstructed
FPGA waveform has 0.046058 relative L2 and 0.005542 RMSE against the CPU WAV,
with no attenuation mixing; these are waveform numerical differences,
separate from spectrum error and perceptual speech quality.

All 42 DPDFNet host tests passed, including eight new audio tests for delayed
reconstruction, first/last samples, tiny recordings, resampling, silence and
invalid inputs. Audio logs and the comparison are saved locally as
`dpdfnet_bin/apex_audio_fpga.log`, `apex_audio_cpu.log` and
`audio_validation_20260913.json`. The existing spectrum runner remains
available for the original benchmark sequence:

```bash
source /home/hunlu/my_torch_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python models/dpdfnet/dpdfnet_compile.py --force
python models/dpdfnet/dpdfnet_sqrt_smoke.py --device rk --dev xdma0
python models/dpdfnet/dpdfnet_state_copy_smoke.py --device rk --dev xdma0
python models/dpdfnet/dpdfnet_gru_smoke.py --device rk --dev xdma0
python models/dpdfnet/dpdfnet_transpose_smoke.py --device rk --dev xdma0

python models/dpdfnet/dpdfnet_run_from_bin.py --device rk --dev xdma0 \
  --input models/dpdfnet/dpdfnet_bin/italy_benchmark_20260913/apex_frames.npy \
  --output models/dpdfnet/dpdfnet_bin/repeat_fpga.npy
python models/dpdfnet/dpdfnet_benchmark_cpu.py --threads 1 \
  --input models/dpdfnet/dpdfnet_bin/italy_benchmark_20260913/apex_frames.npy \
  --output models/dpdfnet/dpdfnet_bin/repeat_cpu.npy
python models/dpdfnet/dpdfnet_compare_outputs.py \
  --reference models/dpdfnet/dpdfnet_bin/repeat_cpu.npy \
  --candidate models/dpdfnet/dpdfnet_bin/repeat_fpga.npy

python models/dpdfnet/dpdfnet_profile.py --device rk --dev xdma0 \
  --input models/dpdfnet/dpdfnet_bin/italy_benchmark_20260913/apex_frames.npy \
  --output models/dpdfnet/dpdfnet_bin/profile.json
```

The profiler restores initial recurrent state for every prefix replay, including
when selecting a later input frame with `--frame`. Use `run_from_bin` for
streaming latency. Rebuild existing v2 bins with `--force` to apply the
optimizations; the streaming ABI is unchanged.

The recorded input is the 5.102-second repository WAV, downmixed and resampled
from 48 kHz stereo to 16 kHz mono, with a 320-sample window and 160-sample hop.
Generated inputs, logs and outputs are local, ignored build products.
`dpdfnet_bin/nan_debug_20260913/verified_regressions.json` records the corrected
baseline. `dpdfnet_bin/optimization_20260913/` contains the archived baseline
and optimized bins, `candidate_regressions.json`, `benchmark_runs.json`,
`benchmark_summary.json`, the hardware smoke logs and `profile_optimized.json`.
Its `benchmark_compare.py` repeats the three alternating runs per artifact.

Environment: host `italy`, Intel Core Ultra 9 285K, RK-256/AXI-256, one core,
333.25 MHz, FPGA stamp `0xb97c477a`, `HW_INFO=0x80214d40`. Python 3.12.3,
PyTorch 2.12.0.dev20260405+cu128, ONNX Runtime 1.30.0, NumPy 2.4.2.
Baseline source was branch `agent/yolov5-queued-config` at `dfcb664`; the
optimized build adds the GRU, row-copy and cropped-transpose changes described
above. Both builds include the NaN and state-alignment corrections.

- ONNX SHA-256: `4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c`.
- Baseline artifact SHA-256: `40701ea2c997c476dc19fa98f69055e598fc14b9d216b83dd36bff423ea1d3ca`.
- Optimized artifact SHA-256: `4f540ae650c27d822f4da24cfb0a0ac7f9cf33e53e84cbd9ce2fc7f1cdd5fc77`.
