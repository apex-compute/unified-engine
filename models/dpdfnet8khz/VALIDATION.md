# Native 8 kHz validation — Italy, 2026-09-14

The official native 8-kHz model runs on RK AXI-256 with finite output and
exact-zero fresh-state silence. Optimized IF8 output matches the corrected
baseline bit for bit across 1,362 frames. Pooled waveform error against CPU
is **1.16% relative L2** across two noisy recordings.

The current deployment measured **9.070 ms FPGA / 9.805 ms host** across the
original 1,362 frames. A sustained 60-second WAV-to-WAV run took **59.210 seconds**
including host audio framing and reconstruction (RTF **0.98684**). It preserved
all 480,000 samples at 8 kHz. Total time from the runner's startup timer was
59.338 seconds; Python imports precede that timer.

Average throughput is below the audio duration on this recording. The strict
10-ms frame deadline is **not yet met**: the audio run had 1,062 late host
frames out of 6,007, with p99 10.222 ms and maximum 29.367 ms. Every FPGA frame
finished within 10 ms. These measurements are on an ordinary Linux host,
with the runner pinned to CPU core 6; they do not guarantee live capture timing.
All current hardware runs set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

Exact artifact hashes, timing summaries and CPU comparisons are in the
[machine-readable report](validation/20260914/report.json).

## Model and hardware

ONNX SHA256:
`6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d`.
Native input is 8 kHz, FFT/window 160, hop 80 samples (10 ms), 81 complex bins,
and 37,860 recurrent-state values. All 492 ONNX operations and state commit
execute in one resident program with one START/HALT per frame. Host STFT/iSTFT
and file I/O are separate. The model's algorithmic delay is 40 ms.

The user updated the board after reset. Current hardware reports build
`0xdf0749de`, matching local Andromeda commit
`df0749de3d4787b51a86a4d0a235e819a6fb52b7`, HW_INFO `0x80214d40`, AXI-256,
one core, 2 GiB DRAM and 333.25 MHz. PCIe access and model execution are now
verified. Current timing uses that detected clock without an override or trace
capture. See the [current hardware receipt](validation/20260914/hardware_current.json).

Earlier results used [Andromeda run 34787221985](https://github.com/apex-compute/andromeda/actions/runs/34787221985),
commit `5fbbfbf095ae7d2f30175b723a4c0a00330ee246`. That bitstream was initially
loaded and verified, then reloaded through JTAG at 02:44 UTC after a reset.
The historical [initial receipt](validation/20260914/hardware_update.json) and
[reload receipt](validation/20260914/hardware_reload_after_reset.json) describe
those events; they are not provenance for the current `df0749de` build.

## Correctness fixes

The inherited convolution policy selected IF4 for twelve dense convolutions.
A first-frame intermediate capture showed about 10.1% error at one pointwise
layer against FP32 weights, but only 0.23% against an IF4-quantized CPU replay.
The native compiler now uses gather-IF8 for every dense convolution;
depthwise and recurrent arithmetic remain BF16.

A separate optimized-transpose problem affected consecutive N=1 matvecs on
this RK build: values entering the first convolution were shifted by one
position. A zero dummy channel keeps that operation on N=2 and restores exact
baseline output. Later layout optimizations also avoid N=1.

The shared zero-sqrt and packed-state boundary fixes are retained. Selection
arithmetic clears or excludes unknown padding because zero times NaN remains
NaN. Silence and speech/silence transitions exercise the recurrent path.

## CPU agreement

The [reference manifest](validation/20260913/cpu_reference_manifest.json)
binds inputs and CPU spectra by SHA256. ONNX Runtime uses one CPU thread on
an Intel Core Ultra 9 285K, averaging **0.642 ms per neural frame**.

| Case | Frames | Spectrum relative L2 | Waveform relative L2 |
| --- | ---: | ---: | ---: |
| p232_007, café noise | 402 | 1.506% | 1.274% |
| p257_018, public-square noise | 448 | 1.187% | 0.821% |
| Fresh-state silence | 128 | Exact zero | — |
| Speech / silence / speech, retained state | 384 | 1.442% | — |

All 1,362 FPGA spectra are finite and match the corrected IF8 baseline exactly.
CPU spectra and state are finite. Waveform comparison uses the same native
STFT/iSTFT, fixed delay compensation and all samples, with no fitted gain,
extra alignment or onset trimming. Pooled waveform RMSE is 0.0006815 across
66,997 samples. This measures agreement with CPU, not denoising quality
against clean speech; broader clean-reference qualification remains pending.

A second corpus alternates the same noisy recordings with 250-ms silence gaps
for 60 seconds, without gain adjustment, and adds fresh-state silence and a
retained-state transition case. Its 6,519 FPGA spectra are finite and pass CPU
agreement screening. The 60-second case has **2.530% spectrum relative L2** and
**2.267% waveform relative L2** against CPU; the one-thread CPU model averages
0.627 ms per frame. This corpus keeps recurrent state across clip boundaries,
so it is a different comparison from the two individually reset recordings.

Both resident state buffers were read once after each sustained spectral run:
all 37,860 logical values were finite. This checks final state, not state on
every frame. The separate WAV-to-WAV run is bit-identical to reconstruction
from saved FPGA spectra. All samples, including the beginning and end, are
included. See the [sustained comparison](validation/20260914/sustained_comparison.json)
and [audio verification](validation/20260914/sustained_waveform_comparison.json).

Source recordings are from VoiceBank-DEMAND. See
[`test_samples/p232_007.README.md`](../../test_samples/p232_007.README.md)
for the repository sample attribution and the reference manifest for hashes.

## Timing

| Measurement | Corrected IF8 baseline | Previous fused IF8 | Current bin |
| --- | ---: | ---: | ---: |
| FPGA build | 5fbbfbf0 | 5fbbfbf0 | df0749de |
| Frames | 1,362 | 1,362 | 1,362 |
| FPGA mean | 18.166 ms | 9.456 ms | 9.070 ms |
| Host mean | 19.147 ms | 10.339 ms | 9.805 ms |
| Host p95 | 19.725 ms | 10.738 ms | 10.079 ms |
| Host p99 | 19.885 ms | 10.972 ms | 10.205 ms |
| Host maximum | 22.687 ms | 13.368 ms | 12.724 ms |
| Host frames above 10 ms | 1,362 | 1,292 | 175 |

The current sustained spectral corpus measured 9.070 ms FPGA / 9.848 ms host
across 6,519 frames, with 1,147 host deadline misses. All frames are counted,
including startup and silence. No warmup frames were discarded. The table's
historical runs differ in FPGA build and host affinity, so they are not an
isolated software speedup experiment. Same-build 64-frame checks confirmed the
two new reshape paths preserve exact output and reduce FPGA time from about
9.296 ms to 9.070 ms.

Host frame timing includes packing, input DMA, START/HALT, output DMA and
output validation. It excludes initial load/upload, STFT/iSTFT and file I/O.
The comparator requires every measured frame to meet the deadline. Prefix
profiling is diagnostic, not a streaming benchmark.

## Deployment sections

The current `streaming-v2` single bin is the artifact used in the original
corpus, sustained corpus and WAV-to-WAV tests above. Its SHA256 is
`905a12a58b6a556386f225bd9b4d5493013a172200d2de6cd875fdd42cf2637d`.
The compiler enables optimized IF8 lowering by default. New coefficient
reshape and 80-column unpadding paths reduce instruction and data movement
while retaining the same neural arithmetic. The unpadding path was verified
on `df0749de` with dirty padding and guarded output buffers.

| Section | Bytes |
| --- | ---: |
| Instructions: 27,118 descriptors | 867,776 |
| Parameter/data section | 22,726,272 |
| Resident image | 23,594,048 |
| Serialized single deployment bin | 23,721,269 |

The parameter/data section includes packed weights, selector constants, initial
state and alignment. It is not the learned parameter count or ONNX file size.
Raw `dpdfnet2_8khz.instructions.bin` and `dpdfnet2_8khz.parameters.bin` sections
were extracted into the ignored deployment directory for inspection. Runtime
execution uses the complete `dpdfnet2_8khz-andromeda.bin` container.

The initial September 13 AXI-512 results were invalid and are excluded.
Their [historical report](validation/20260913/report.json) is retained; the
runner now rejects mismatched AXI width before reset or upload.

A later experimental host DMA optimization used positional vector I/O and
triggered a kernel BUG in `do_iter_readv_writev` / `do_pwritev` on the installed
XDMA driver. It is not part of the production runner or accepted measurements.
That vector-I/O path has been removed. The current runner caches DMA handles
and uses only scalar seek/read/write transfers, with cleanup on success and
failure. Adaptive HALT polling was also tested and rejected because it worsened
host timing; the established polling method remains in production.

Host verification: **89 native 8-kHz tests and 63 existing DPDFNet tests pass**.
