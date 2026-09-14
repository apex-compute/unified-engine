# Noisy speech tests longer than 20 seconds

Five test sequences cover bus, café, living-room, office and public-square
noise. Each has a noisy input and enhanced outputs from both deployed FPGA
models. All WAVs are mono FLOAT at 16 kHz and preserve the complete input
sample count. The 8-kHz runner resamples audio on the host before inference,
then restores the source rate for playback; this does not restore
high-frequency information.

| Environment | Duration | Noisy input | 8-kHz model FPGA output | 16-kHz model FPGA output |
| --- | ---: | --- | --- | --- |
| Bus | 23.682 s | [WAV](noisy/bus.wav) | [WAV](fpga8k/bus.wav) | [WAV](fpga16k/bus.wav) |
| Café | 20.509 s | [WAV](noisy/cafe.wav) | [WAV](fpga8k/cafe.wav) | [WAV](fpga16k/cafe.wav) |
| Living room | 20.625 s | [WAV](noisy/living.wav) | [WAV](fpga8k/living.wav) | [WAV](fpga16k/living.wav) |
| Office | 21.980 s | [WAV](noisy/office.wav) | [WAV](fpga8k/office.wav) | [WAV](fpga16k/office.wav) |
| Public square | 21.473 s | [WAV](noisy/psquare.wav) | [WAV](fpga8k/psquare.wav) | [WAV](fpga16k/psquare.wav) |

These are joined test sequences from the existing VoiceBank-DEMAND mixtures,
not continuous new recordings. Each includes four complete source utterances
covering nominal SNRs 2.5, 7.5, 12.5 and 17.5 dB. The shortest utterance is
repeated once to reach at least 20 seconds. Joins contain 250 ms of zero
silence. Complete utterances are retained, with no gain adjustment, crossfade,
truncation or newly mixed noise. Each source is resampled from 48 kHz directly
to 16 kHz before concatenation. Recurrent state continues across every join.

The [input manifest](input_manifest.json) records the exact source hashes,
segment positions, repetition, silence gaps and sample counts. Each FPGA
output has a matching `.metrics.json` and `.log` next to it. Those files bind
input/output/bin hashes, live hardware, all frame timings and transfer counts.
The [results](results.json) summarize timing and comparisons with each model's
own CPU reference. Waveform error measures numerical agreement; it is not a
speech-quality score. The existing 16-kHz model's documented accuracy issue
remains open.

## FPGA execution

These files use the unchanged single-bin deployments on Italy RK-256,
build `0xdf0749de`, 333.25 MHz. The runner uses one host math thread and CPU
core 6. Each file starts with fresh model state. Program, parameter/data and
initial state are uploaded to DRAM once; each 10-ms hop uses one input write,
one execute command, HALT and one output read. All neural operations execute
on FPGA, with state retained in DRAM. Host STFT/iSTFT handles audio framing
and reconstruction, including flushing and compensating the model's 40-ms
delay. See the [single-bin benchmark](../../DUAL_RATE_BENCHMARK.md) for the
full execution contract and instruction/parameter sections.

RTF is processing time divided by audio duration: below 1 is faster than
realtime on average. Per-frame deadline misses are recorded separately;
average realtime throughput does not guarantee every 10-ms deadline.

## Measured results

RTF includes host framing/reconstruction and WAV I/O, excluding one-time
model loading and initialization. All frames are included, including flush
frames, with no warmup discarded. CPU references use each model’s pinned
FP32 ONNX on one thread, CPU core 7.

| Environment | 8-kHz audio RTF | 16-kHz audio RTF | 8-kHz error vs CPU | 16-kHz error vs CPU |
| --- | ---: | ---: | ---: | ---: |
| Bus | 0.987940 | 1.435041 | 1.022% | 3.681% |
| Café | 0.990932 | 1.434937 | 1.908% | 4.012% |
| Living room | 0.988341 | 1.437744 | 1.541% | 3.090% |
| Office | 0.989439 | 1.437438 | 0.924% | 2.913% |
| Public square | 0.992376 | 1.436649 | 1.199% | 5.474% |

Across **108.2705 seconds and 10,860 frames per model**:

| Measurement | Native 8 kHz | Existing 16 kHz |
| --- | ---: | ---: |
| Audio processing time | 107.162600 s | 155.513434 s |
| Audio RTF | 0.989767 | 1.436342 |
| CPU audio RTF | 0.067306 | 0.068761 |
| Mean FPGA time / frame | 9.069716 ms | 13.447038 ms |
| Mean host time / frame | 9.849806 ms | 14.308849 ms |
| Host p99 / maximum | 10.660048 / 15.046639 ms | 14.815659 / 21.584982 ms |
| Host frames over 10 ms | 1,699 / 10,860 | 10,860 / 10,860 |
| FPGA frames over 10 ms | 0 / 10,860 | 10,860 / 10,860 |
| Pooled waveform error vs CPU | 1.341% | 3.983% |

The 8-kHz model is faster than realtime on average for every file, with
little headroom and host frame deadline misses. The 16-kHz model is slower
than realtime. All ten outputs are finite; every source sample is retained.
Error is relative L2 over the complete waveform, without fitted gain,
alignment or trimming. The model’s fixed delay compensation is applied
equally to CPU and FPGA output. Lower error on these joined sequences does
not resolve the 16-kHz accuracy issue observed in earlier individual clips.

The [summary](summary.json), [full comparison](results.json), and
[independent audit](independent_audit.json) bind these results to the raw
logs, metrics and WAV hashes. The [8-kHz](cpu8k_reference_manifest.json) and
[16-kHz CPU manifests](cpu16k_reference_manifest.json) preserve the reference
model, waveform, state and timing hashes; their generated files remain in
the local validation paths recorded in the reports.

## Reproduce

To run either model on a provided input from the repository root:

```bash
source /home/hunlu/my_torch_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --device rk --dev xdma0 --cpu-core 6 \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus.wav \
  --output /tmp/bus_fpga8k.wav

python models/dpdfnet/dpdfnet_run_from_bin.py \
  --device rk --dev xdma0 --cpu-core 6 \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus.wav \
  --output /tmp/bus_fpga16k.wav
```

Use the normal hardware lock to exclude concurrent FPGA jobs. The default
bin paths are the same artifacts identified in the per-run reports.
Generated CPU reference WAVs and clean composites remain local validation
artifacts; the noisy inputs and actual FPGA output WAVs are checked in here.

The [SHA-256 inventory](SHA256SUMS) covers every file in this collection.
Verify it from this directory with `sha256sum -c SHA256SUMS`.

## Source attribution

Cassia Valentini-Botinhao (2017), University of Edinburgh/CSTR, *Noisy speech
database for training speech enhancement algorithms and TTS models, 2016*,
[DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117), CC BY 4.0.
Environmental noise recordings originate from [DEMAND](https://zenodo.org/records/1227121),
Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent (2013), CC BY-SA 3.0.
These files are adaptations through resampling, concatenation and, for the
FPGA outputs, speech enhancement. Original archive metadata, licensing links
and all source hashes are retained in the input manifest and the repository's
[noisy-test documentation](../../NOISY_TESTS.md).
