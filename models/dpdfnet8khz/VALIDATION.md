# Native 8 kHz validation — Italy, 2026-09-13

**CPU inference works. FPGA accuracy and real-time performance are not yet
verified.** The required target is RK AXI-256 with queue-CONFIG convolution.
The device available during the first 8 kHz smoke instead reported AXI-512,
build `0x0305d87d`. No timing from that run is accepted as model performance.

## Verified software behavior

The official ONNX model is pinned to
`6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d`.
It consumes 81 complex spectrum bins per 80-sample / 10-ms hop at 8 kHz,
and maintains 37,860 state values. All 492 graph operations compile offline,
followed by state commit and a single HALT.

The CPU reference corpus contains 1,362 frames: two noisy speech recordings,
128 silence frames, and a 384-frame speech/silence/speech transition. Every
CPU spectrum and recurrent state is finite; fresh-state silence produces
exact zero. Mean neural inference time is **0.642 ms/frame**, with one ONNX
Runtime thread. This is CPU performance, not an FPGA estimate.

A separate audio-to-audio run on `test_samples/p232_007.wav` processed its
3.956-second recording in 0.323 seconds including model session loading,
audio conversion and file I/O. Neural inference averaged 0.630 ms/frame.
Output is finite mono audio with the original 48-kHz rate and all 189,883
samples preserved; the neural model itself runs at 8 kHz. This single-file
throughput check does not establish a hard deadline guarantee.

All **49 native 8-kHz host tests and 63 existing 16-kHz regression tests pass**.
Host tests cover metadata/hash rejection, state initialization, audio endpoints,
silence, unaligned 80/81-bin layouts, reflect padding, PixelShuffle, state
copy boundaries, optimized convolution layouts and rejection of AXI-512 before
upload. The CPU reference preparer reproduced all 1,362 saved input and CPU
output frames bit-for-bit in a separate CLI run.

## Compiled deployment sizes

| Quantity | Initial build | Opt-in optimized build |
| --- | ---: | ---: |
| Instructions | 53,890 | 33,342 |
| Instruction bytes | 1,724,480 | 1,066,944 |
| Parameter/data bytes | 18,077,824 | 18,077,824 |
| Resident image bytes | 19,802,304 | 19,144,768 |
| Serialized single-bin bytes | 19,929,077 | 19,271,634 |

The optimization removes 38.1% of instruction descriptors through batched
state copies, native coefficient reshaping and reduced convolution transpose
work. It remains opt-in until hardware output comparison passes. Parameter/data
includes packed weights, selector constants, initial state and alignment; it is
not the learned parameter count or ONNX file size. The single-bin container
also includes deployment metadata. Exact hashes are in
[the machine-readable report](validation/20260913/report.json).

## Why FPGA timing is pending

Earlier 16-kHz tests used build `0xb97c477a`, AXI-256, at 333.25 MHz. The new
device reports build `0x0305d87d`, HW_INFO `0xa0214d40`, AXI-512 at the same
clock. The native 8-kHz program was compiled for AXI-256; some DMA descriptors
use 32-byte boundaries that do not satisfy AXI-512 alignment.

Before the new compatibility check was added, a 64-frame run returned identical
output for changing inputs, with magnitudes up to `1.56e38`. The previously
tested 16-kHz bin also returned identical output across 16 changing frames on
this build. These results are invalid; the exact reason execution failed on
the different firmware is not established. The runner now checks the compiled
width before resetting or writing to the device.

## Remaining acceptance checks

1. Run initial and optimized bins on compatible RK-256 hardware, compare against
   CPU spectra and reconstructed WAVs, and verify that optimization preserves
   baseline output. Include silence and speech transitions to check NaN behavior.
2. Profile the correct native graph, then optimize measured bottlenecks.
3. Measure sustained host and FPGA frame times under normal load. Report mean,
   p95, p99, maximum and all 10-ms deadline misses, plus audio framing overhead.
4. Evaluate the full noisy speech corpus against CPU and clean references before
   claiming acceptable speech quality. A coarse numerical screening threshold
   only catches obvious execution failures.

At 8 kHz, the processing budget remains **10 ms per hop**. The model's 40-ms
algorithmic delay is separate. Fewer samples or instructions alone do not prove
that the RK-256 implementation meets this requirement.

The noisy speech sources and hashes are recorded in
[the CPU reference manifest](validation/20260913/cpu_reference_manifest.json).
The recordings come from the VoiceBank-DEMAND evaluation set; attribution and
license information are in `test_samples/README.md` and `test_samples/LICENSE`.
