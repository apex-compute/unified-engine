# Native 8 kHz and 16 kHz DPDFNet2 — Italy, 2026-09-14

Each deployed model runs its complete neural graph from its own single bin,
with one input DRAM write, one START/HALT and one output DRAM read per 10-ms
hop. The 8-kHz deployment achieves average realtime throughput on the measured
60-second recording. It still misses individual host deadlines. The existing
16-kHz deployment is slower than realtime and retains a substantial numerical
disagreement with its CPU reference on some samples.

The board reports `0xdf0749de`, RK AXI-256, 333.25 MHz. Both models were tested
sequentially on the same programmed board, with scalar DMA handle reuse and the host runner
pinned to CPU core 6. CPU references use ONNX Runtime FP32 with one thread on
core 7 of Italy's Intel Core Ultra 9 285K. `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1` are set. No trace capture or clock override is enabled.

## Real-time ratio

**RTF = processing seconds / audio seconds.** An RTF below 1 means faster
than realtime on average; `1 / RTF` is throughput expressed as times realtime.
This does not establish that every frame meets its deadline.

Both models processed the same 60-second source file, a 16-kHz composite of
the original noisy recordings with 250-ms silence gaps. The 8-kHz frontend
resamples it to its native rate. The source preserves the higher bandwidth
needed for the 16-kHz test; it was not made by upsampling 8-kHz audio.

| Measured 60-second WAV-to-WAV run | Native 8 kHz | Existing 16 kHz |
| --- | ---: | ---: |
| New samples per 10-ms hop | 80 | 160 |
| FPGA execution mean | 9.070 ms | 13.447 ms |
| Host frame mean, including FPGA and transfers | 9.848 ms | 14.300 ms |
| Audio processing, including host framing/reconstruction | 59.216 s | 85.956 s |
| Audio RTF | **0.98693** | **1.43260** |
| Throughput relative to realtime | **1.01325×** | **0.69803×** |
| Total measured runner time, including loading/initialization | 59.363 s | 86.126 s |
| One-thread CPU audio processing | 4.008 s | 4.113 s |
| CPU audio RTF | 0.06680 | 0.06855 |
| Host p95 / p99 | 10.119 / 10.225 ms | 14.613 / 15.300 ms |
| Host maximum | 13.500 ms | 18.250 ms |
| Host frames exceeding 10 ms | **1,099 / 6,007** | **6,007 / 6,007** |
| FPGA frames exceeding 10 ms | 0 / 6,007 | 6,007 / 6,007 |

All 6,007 frames are counted, including end-of-file flush frames. Python imports
precede the runner's startup timer. Audio processing time excludes one-time
loading/upload/initialization; the total-time row includes them. Both output
files retain all 960,000 source samples at 16 kHz. The native 8-kHz model's
output is upsampled for this source-rate file; that does not restore frequencies
above its native audio bandwidth. Native-rate listening files are also provided.

The wider spectral corpus contains 7,369 frames per model: the two original
clips, the 60-second composite, 128 silent frames and a 384-frame retained-state
speech/silence/speech sequence. No warmup frames are dropped.

| All 7,369 spectral frames | Native 8 kHz | Existing 16 kHz |
| --- | ---: | ---: |
| FPGA mean / neural RTF | 9.070 ms / 0.90697 | 13.447 ms / 1.34471 |
| Host mean / neural RTF | 9.847 ms / 0.98467 | 14.304 ms / 1.43039 |
| CPU neural mean / RTF | 0.652 ms / 0.06516 | 0.669 ms / 0.06687 |
| Host frames exceeding 10 ms | 1,349 | 7,369 |

These frame timers exclude audio framing, reconstruction, file I/O and initial
upload. CPU neural time measures ONNX Runtime execution; FPGA host time also
includes transfers and completion waiting. The WAV-to-WAV table supplies the
more complete audio processing measurement. These are recorded-file tests,
not live microphone capture or hard realtime guarantees.

## Verified DRAM execution sequence

1. Load and validate one `.bin` container. Upload its complete resident image
   to DRAM once: packed parameters/constants, initial recurrent state and the
   instruction program are contiguous sections of that image.
2. For each new 10-ms audio hop, host STFT forms the input spectrum using the
   overlapping 20-ms window. Upload that padded BF16 spectrum to input DRAM.
3. Issue one execute command pointing at the resident program. The FPGA runs
   all graph operations and commits recurrent state, ending at its sole HALT.
4. After HALT and queue idle, read one enhanced spectrum from output DRAM.
   Host iSTFT reconstructs audio. Continue with the next hop and resident state.

The program and parameters are **not reuploaded each frame**; reloading initial
state each hop would break the recurrent model. The supplied neural graph uses
spectra at its boundary. Audio framing/reconstruction remain on the host, with
no CPU neural operations or intermediate host tensor transfers. Both models
have a four-hop, 40-ms model delay, separate from execution time.

The [hardware contract audit](validation/20260914_dual_rate/frame_contract.json)
observed the actual driver DMA/start calls and HALT/queue registers for two
frames per model. It checked the complete upload hash, exact transfer addresses
and byte counts, event order, and one terminal HALT with no SWIs. This is driver
API/register evidence rather than a PCIe bus trace. The full audio-run logs
also record exactly one model upload and 6,007 input writes, kicks, HALTs and
output reads, with zero intermediate host transfers, for each model.

| Resident-bin contract | Native 8 kHz | Existing 16 kHz |
| --- | ---: | ---: |
| ONNX operations + state commit | 492 + 1 | 472 + 1 |
| Logical spectrum shape | `[1,1,81,2]` | `[1,1,161,2]` |
| Input / output DRAM transfer per frame | 10,368 / 10,368 bytes | 20,608 / 20,608 bytes |
| Instructions, including HALT and alignment NOPs | 27,118 | 40,264 |
| Raw instruction section | 867,776 bytes | 1,288,448 bytes |
| Raw parameter/data section | 22,726,272 bytes | 16,575,360 bytes |
| Complete resident DRAM image | 23,594,048 bytes | 17,863,808 bytes |
| Serialized single bin on disk | 23,721,269 bytes | 17,984,019 bytes |

Parameter/data size includes packed weights, scales, selector constants,
initial state and alignment; it is not the learned parameter count. Raw
`.instructions.bin` and `.parameters.bin` files were extracted beside each
deployment for inspection. The runner consumes the complete single-bin
container. Exact section hashes and DRAM addresses are recorded in
[deployment_sections.json](validation/20260914_dual_rate/deployment_sections.json).

## CPU agreement and listening samples

Every measured FPGA output is finite; fresh-state silence is exactly zero.
The independent CPU references also have finite output and recurrent state.
Both separate 60-second audio runs match reconstruction from their saved FPGA
spectra bit for bit, including the first and last samples.

Waveform error below compares each FPGA with its own pinned FP32 CPU model,
using all source-rate samples, fixed model-delay compensation and no fitted
alignment, fitted gain, normalization or trimming. It measures numerical
agreement, not denoising quality against clean speech.

| Source recording | Duration | Native 8 kHz waveform relative L2 | Existing 16 kHz waveform relative L2 |
| --- | ---: | ---: | ---: |
| `p232_007`, café noise | 3.956 s | **1.262%** | **11.218%** |
| `p257_018`, public-square noise | 4.419 s | **0.821%** | **24.768%** |
| Matched sustained composite | 60.000 s | **1.890%** | **9.259%** |

The existing 16-kHz deployment also exceeds the 25% coarse spectrum-error
screen on `p257_018` and the transition sequence. Finite output does not make
those accuracy checks pass. Its previously identified onset problem remains
an open issue; this task adds measurement/transport parity without changing
that model's neural arithmetic.

The [decoded instruction audit](validation/20260914_dual_rate/precision_audit.json)
finds 15 IF8 dense convolutions in the 8-kHz bin, versus 12 IF4 and 3 IF8 in
the 16-kHz bin. Both use BF16 activations/recurrent arithmetic. This is a
comparison of the current deployments, with different optimizations and
precision policies; their speed or accuracy difference cannot be attributed
to sample rate alone.

The checked-in [listening samples](validation/20260914_dual_rate/samples/README.md)
include noisy, CPU and FPGA audio for both recordings at each model's native
rate. Their hashes and sample counts are in
[listening_samples.json](validation/20260914_dual_rate/listening_samples.json).

A limited clean-reference check on `p257_018`, evaluating all outputs in the
common 8-kHz band, gives PESQ-NB 3.670 noisy, 4.127 FPGA-8k and 3.638 FPGA-16k;
STOI is 0.98837, 0.98838 and 0.96207 respectively. This is one narrowband sample,
not a full-band or corpus-wide quality ranking. CPU scores, source provenance
and the exact method are in
[quality_sample_comparison.json](validation/20260914_dual_rate/quality_sample_comparison.json).

## Run from the existing bins

From the repository root on Italy, using the existing Python environment:

```bash
source /home/hunlu/my_torch_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --bin models/dpdfnet8khz/dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin \
  --device rk --dev xdma0 --cpu-core 6 \
  --input test_samples/p232_007.wav --output /tmp/p232_8k_fpga.wav

python models/dpdfnet/dpdfnet_run_from_bin.py \
  --bin models/dpdfnet/dpdfnet_bin/dpdfnet2-andromeda.bin \
  --device rk --dev xdma0 --cpu-core 6 \
  --input test_samples/p232_007.wav --output /tmp/p232_16k_fpga.wav
```

Each CLI invocation loads once and internally repeats one execute/readback
cycle per hop. Output WAVs retain the source rate and sample count; select an
8-kHz/16-kHz input file when a native-rate output file is desired. The bin paths
shown are also the defaults. `--cpu-core` is optional and selects an available
host core; pinning alone does not guarantee a deadline.

To reproduce the diagnostic contract check while excluding concurrent FPGA jobs:

```bash
flock /tmp/pcie_ci_hw_italy.lock python models/dpdfnet/dpdfnet_audit_frame_contract.py \
  --input test_samples/p232_007.wav --output /tmp/dpdfnet_frame_contract.json
```

If deployment bins are absent, build them first using the respective
`dpdfnet8khz_compile.py --download --force` and `dpdfnet_compile.py --force`
commands described in each model README. Compilation is a separate offline
step; these hardware runners do not compile or run ONNX inference.

Host validation passes **89 native 8-kHz tests and 71 existing 16-kHz tests**.
The [report manifest](validation/20260914_dual_rate/manifest.json) binds current
measurements, CPU references, deployment sections, listening samples and the
contract audit by hash. Generated model bins and full-length recordings remain
local build products.
