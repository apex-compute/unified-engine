# Native 8 kHz validation — Italy, 2026-09-14

The official native 8-kHz model runs on RK AXI-256 with finite output and
exact-zero fresh-state silence. Optimized IF8 output matches the corrected
baseline bit for bit across 1,362 frames. Pooled waveform error against CPU
is **1.16% relative L2** across two noisy recordings.

The latest completed full-corpus run measured **9.456 ms FPGA** and
**10.339 ms host time** per 10-ms frame. FPGA execution met the deadline on
all frames; host overhead still caused misses. A subsequent 64-frame reduction
optimization measured **9.296 ms FPGA / 10.123 ms host**, with identical output.
The full end-to-end real-time requirement is **not yet met**.

Exact artifact hashes, timing summaries and CPU comparisons are in the
[machine-readable report](validation/20260914/report.json).

## Model and hardware

ONNX SHA256:
`6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d`.
Native input is 8 kHz, FFT/window 160, hop 80 samples (10 ms), 81 complex bins,
and 37,860 recurrent-state values. All 492 ONNX operations and state commit
execute in one resident program with one START/HALT per frame. Host STFT/iSTFT
and file I/O are separate. The model's algorithmic delay is 40 ms.

Italy was programmed from [Andromeda run 34787221985](https://github.com/apex-compute/andromeda/actions/runs/34787221985),
commit `5fbbfbf095ae7d2f30175b723a4c0a00330ee246`. After the initial manual
rescan, hardware reported build `0x5fbbfbf0`, HW_INFO `0x80214d40`, AXI-256,
one core, 2 GiB DRAM, and 333.25 MHz. Timing uses the detected clock without
an override or trace capture. Configuration was loaded through JTAG, not
written to flash. See the [hardware receipt](validation/20260914/hardware_update.json).

After a later host reset, the same bitstream was reloaded at 02:44 UTC.
Post-program PCIe verification is pending another manual rescan. Earlier
performance results above were obtained before that reset.

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

Source recordings are from VoiceBank-DEMAND. See
[`test_samples/p232_007.README.md`](../../test_samples/p232_007.README.md)
for the repository sample attribution and the reference manifest for hashes.

## Timing

| Full-corpus measurement | Corrected IF8 baseline | Fused IF8 |
| --- | ---: | ---: |
| FPGA mean | 18.166 ms | 9.456 ms |
| Host mean | 19.147 ms | 10.339 ms |
| Host p95 | 19.725 ms | 10.738 ms |
| Host p99 | 19.885 ms | 10.972 ms |
| Host maximum | 22.687 ms | 13.368 ms |
| Host frames above 10 ms | 1,362 / 1,362 | 1,292 / 1,362 |

Host frame timing includes packing, input DMA, START/HALT, output DMA and
output validation. It excludes initial load/upload, STFT/iSTFT and file I/O.
The comparator requires every measured frame to meet the deadline. Prefix
profiling is diagnostic, not a streaming benchmark.

## Deployment sections

The current `streaming-v2` single-bin image is byte-identical to the tested
64-frame candidate; the version/metadata change forces older native artifacts
to be rebuilt. The compiler enables the optimized IF8 lowering by default.

| Section | Bytes |
| --- | ---: |
| Instructions: 28,072 descriptors | 898,304 |
| Parameter/data section | 22,906,496 |
| Resident image | 23,804,800 |
| Serialized single deployment bin | 23,931,957 |

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
That vector-I/O path has been removed. The established scalar transfer path
remains the runner default; the revised handle-caching candidate is not enabled.
