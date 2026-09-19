# DPDFNet2 tests with recorded environmental noise

The [current noisy-audio report](validation/20260914_noisy20s/README.md) covers
eight sequences over 20 seconds, including four additional 2.5-dB SNR tests,
both models’ FPGA WAVs, and KU5P resources. The historical 20-clip evaluation
below is unchanged.

On 2026-09-13, Italy RK-256 processed 20 paired noisy-speech test clips and
two clean-speech controls through the production WAV-to-WAV bin runner.
All 9,890 frames completed with finite output. Average quality improved,
but the test exposed an existing FPGA speech-onset accuracy problem and
several cases where enhancement reduces an objective quality score.

## Test coverage and sources

The fixed subset uses the original [VoiceBank-DEMAND test recordings](https://datashare.ed.ac.uk/handle/10283/2791):
two speakers (`p232`, `p257`), five environments and four nominal SNR levels
(2.5, 7.5, 12.5 and 17.5 dB). There is one clip per environment/SNR condition,
ten clips per speaker, totaling 85.687 seconds. Selection was fixed before
scoring, with original 3–10 second clips closest to five seconds per condition.
The precise selection and hashes are pinned in [noisy_test_cases.json](noisy_test_cases.json).

The environments are a public transit bus, café terrace, living room, small
office and public town square. These are the dataset's mixtures of recorded
speech and recorded environmental noise. They are not simultaneous live
microphone captures of speakers in those environments. No new noise was
synthesized or mixed for this evaluation.

Attribution: Cassia Valentini-Botinhao (2017), University of Edinburgh/CSTR,
*Noisy speech database for training speech enhancement algorithms and TTS
models, 2016*, [DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117), CC BY 4.0.
The original environmental recordings are from [DEMAND](https://zenodo.org/records/1227121),
Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent (2013), CC BY-SA 3.0.
The pinned manifest records both attributions and the original license URLs.

## Method

Original paired mono WAVs are verified by SHA-256 and resampled from 48 to
16 kHz, once each, without gain normalization. The noisy WAV is fed to both
the FPGA bin runner and the pinned FP32 ONNX CPU reference. Each clip starts
with fresh recurrent state. The shared audio wrapper flushes and compensates
the model's four-hop delay, preserving the complete waveform. All neural
operations in the FPGA run execute on the FPGA, with one START/HALT per frame.

Scores compare the full output waveform with the clean reference, using
the same clean/noisy pairs for every backend. No additional alignment or
gain fitting is applied before scoring. The standard metric libraries retain
their own internal processing. We report mean-removed [SI-SDR](https://arxiv.org/abs/1811.02508),
standard [STOI](https://github.com/mpariente/pystoi) and wideband
[PESQ](https://github.com/ludlows/PESQ). Higher is better for all three.
The dataset's nominal SNR labels are retained; SI-SDR is measured separately.

The bin is unchanged from the optimized timing benchmark:
`4f540ae650c27d822f4da24cfb0a0ac7f9cf33e53e84cbd9ce2fc7f1cdd5fc77` (SHA-256).
Hardware: RK-256, `xdma0`, FPGA stamp `0xb97c477a`, 333.25 MHz. Evaluation
packages: `pystoi==0.4.1`, `pesq==0.0.4`. The CPU reference runs on Italy's
Intel Core Ultra 9 285K with ONNX Runtime 1.30.0, `CPUExecutionProvider`,
and one intra-op/inter-op thread. Its pinned FP32 ONNX SHA-256 is
`4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c`.

## Results

Every mean below contains all 20 noisy clips. The two clean controls and
the additional user-provided clip are excluded from these averages.

| Signal | SI-SDR (dB) | STOI | PESQ-WB |
|---|---:|---:|---:|
| Noisy input | 8.989 | 0.9207 | 2.0625 |
| FP32 CPU output | 19.506 | 0.9298 | 2.8079 |
| RK-256 FPGA output | 18.693 | 0.9284 | 2.7356 |
| FPGA improvement over noisy input | +9.704 | +0.0078 | +0.6730 |

FPGA SI-SDR improves on 19/20 clips, PESQ on 17/20, and STOI on 10/20.
STOI decreases on the other ten clips, so these results do not support a
claim of consistent intelligibility improvement. The CPU model also has
STOI regressions on nine clips and PESQ regressions on two.

### Comparison with the FP32 CPU model

These comparisons pair CPU and FPGA runs of the same 20 clips. Quality
scores on both sides use the clean speech as their reference; the difference
column below is FPGA minus CPU. All 20 pairs are present for every metric.

| Quality metric | CPU mean | FPGA mean | Mean difference | FPGA lower than CPU |
|---|---:|---:|---:|---:|
| SI-SDR (dB) | 19.506 | 18.693 | -0.813 | 20/20 |
| STOI | 0.9298 | 0.9284 | -0.0014 | 11/20 |
| PESQ-WB | 2.8079 | 2.7356 | -0.0723 | 17/20 |

Direct comparison of the saved FPGA waveform against the CPU waveform gives
**5.09% pooled relative L2**, **2.81% median per-clip relative L2**, and
0.003415 pooled RMSE. Pooling means summing squared errors and CPU signal
energies across clips before taking their ratio; it is not the mean of
per-clip percentages. Waveforms are compared without realignment, gain
adjustment or resampling. The largest difference is `p257_018`: **24.77%
relative L2** and a **9.372 dB SI-SDR deficit** against CPU. That case remains
included in the averages and is discussed below.

The logged timings over the same 8,700 noisy-clip frames are weighted by
frame count, including the end-of-file flush frames:

| Execution measurement | ms/frame | RTF for a 10 ms hop |
|---|---:|---:|
| CPU, ONNX Runtime FP32, one thread | 0.650 | 0.065 |
| RK-256 FPGA execution counter | 13.447 | 1.345 |
| RK-256 including host frame handling/transfers | 14.382 | 1.438 |

The FPGA execution counter measures **20.70 times the CPU inference time**
on these runs. The CPU timer covers ONNX Runtime inference; the FPGA host
timer additionally includes packing, transfers and waiting. One-time model
loading/upload, STFT/iSTFT and file I/O are excluded. These are the actual
quality-suite runs, not a separate repeated timing campaign.

An independent audit verified all 46 CPU/FPGA WAV hashes and runtime logs
across the 20 noisy cases, two clean controls and additional user sample.
All 138 quality scores were recomputed; STOI/PESQ match exactly and SI-SDR
differs by at most 2.5e-14 dB from floating-point arithmetic.

The checked-in [CPU_COMPARISON.csv](CPU_COMPARISON.csv) retains all 23 paired
cases, including quality deltas, waveform errors, timings and output hashes.
Its `subset` column distinguishes the 20 `noisy_test` cases used in the means
from the two `clean_control` cases and one `additional_sample`. Relative L2
is stored as a fraction, and per-case timings are total seconds.

Each environment row averages its four SNR conditions:

| Environment | Noisy SI-SDR | FPGA SI-SDR | Noisy STOI | FPGA STOI | Noisy PESQ | FPGA PESQ |
|---|---:|---:|---:|---:|---:|---:|
| Bus | 9.270 | 19.879 | 0.9568 | 0.9650 | 2.570 | 3.000 |
| Café | 9.254 | 17.752 | 0.9107 | 0.9302 | 1.621 | 2.371 |
| Living room | 9.397 | 17.736 | 0.9405 | 0.9504 | 1.613 | 2.587 |
| Office | 8.068 | 21.594 | 0.8687 | 0.8566 | 2.574 | 3.000 |
| Public square | 8.956 | 16.501 | 0.9268 | 0.9400 | 1.934 | 2.719 |

### Reproducible accuracy failure

`p257_018`, public-square noise at nominal 17.5 dB, gets worse on the FPGA:
SI-SDR falls from **16.830 dB noisy to 11.614 dB enhanced**; CPU reaches
20.987 dB. FPGA STOI is 0.9621 versus noisy 0.9884, and PESQ is 2.6407
versus noisy 2.7852. These values are retained in all aggregate results.

The optimized repeat and the archived pre-optimization bin produce
bit-identical waveforms on this case. The issue therefore predates the
software optimizations. There are no NaNs or clipping. Diagnostic comparison
with CPU places 99.23% of the difference energy at the speech onset,
approximately 0.59–0.80 seconds, where FPGA output is attenuated too strongly.
The best diagnostic lag is zero samples; no scores were realigned. The
operator-level cause remains unresolved.

The subsequent [accuracy assessment and binary report](dpdfnet_bin/bin_report_20260913/README.md)
includes CPU counterfactuals, reproducible diagnostic code, and verified raw
instruction/parameter-data extracts. Rounding CPU constants/state to BF16 gives
2.35% waveform error on this clip, versus 24.77% for the FPGA. Omitting
LayerNorm epsilon alone gives 0.0037%; neither experiment establishes the root
cause, because they do not emulate all FPGA arithmetic.

Other FPGA PESQ regressions are `p257_324` (bus, 17.5 dB) and `p257_015`
(office, 12.5 dB); both also regress on CPU. The two clean-only controls
remain finite but lose quality: FPGA PESQ is 3.561 and 3.313 versus 4.644
for unchanged clean speech. Enhancement is not harmless on already-clean
audio.

### User sample: p232_007.wav

`test_samples/p232_007.wav` exactly matches the original dataset's noisy
café clip at nominal 12.5 dB by SHA-256. It was evaluated separately against
its matching clean reference:

| Signal | SI-SDR (dB) | STOI | PESQ-WB |
|---|---:|---:|---:|
| Noisy input | 11.810 | 0.9370 | 1.5561 |
| FP32 CPU output | 18.179 | 0.9564 | 2.6366 |
| RK-256 FPGA output | 15.316 | 0.9558 | 2.5953 |

The output at the original 48 kHz rate is
`dpdfnet_bin/p232_007_enhanced_fpga.wav`, mono with all 189,883 original
samples. A matching CPU run is saved as
`dpdfnet_bin/p232_007_enhanced_cpu.wav`. At the 16 kHz scoring rate, the FPGA
waveform differs from CPU by 11.22% relative L2; quality deltas are -2.864 dB
SI-SDR, -0.000623 STOI and -0.0413 PESQ. The original noisy input is included
unchanged under `test_samples/`, with its source attribution and license.

## Reproduce and listen

From the repository root on Italy:

```bash
source /home/hunlu/my_torch_env/bin/activate
python -m pip install -r models/dpdfnet/requirements-eval.txt
python models/dpdfnet/dpdfnet_prepare_noisy.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python models/dpdfnet/dpdfnet_evaluate_noisy.py \
  --manifest models/dpdfnet/dpdfnet_bin/noisy_eval_20260913/sources/cases.json \
  --output-dir models/dpdfnet/dpdfnet_bin/noisy_eval_20260913/results \
  --device rk --dev xdma0 --clean-controls
```

Preparation fetches only the selected original archive members and verifies
ZIP size/CRC, WAV format and pinned SHA-256. Existing downloads are verified
and reused. `--limit 1` on both preparation and evaluation gives a small smoke
test; running preparation again without the limit restores the full manifest.
Use `--backends cpu` for a CPU-only evaluation.

Recompute the CPU comparison from existing saved runs, without rerunning the
model, and export per-case JSON/CSV:

```bash
python models/dpdfnet/dpdfnet_compare_cpu.py \
  --results models/dpdfnet/dpdfnet_bin/noisy_eval_20260913/results
```

The comparison verifies recorded output hashes before reading the paired
waveforms. It excludes clean controls from the aggregate while retaining
their per-case results, and reports valid-pair counts and missing data.

Run the included noisy sample through both models at its original audio rate:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python models/dpdfnet/dpdfnet_run_cpu.py \
  --input test_samples/p232_007.wav \
  --output models/dpdfnet/dpdfnet_bin/p232_007_enhanced_cpu.wav
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python models/dpdfnet/dpdfnet_run_from_bin.py \
  --device rk --dev xdma0 --input test_samples/p232_007.wav \
  --output models/dpdfnet/dpdfnet_bin/p232_007_enhanced_fpga.wav
```

Each result directory contains `clean.wav`, `noisy.wav`, `cpu.wav`,
`fpga.wav` and the actual runner logs. The local
[listening index](dpdfnet_bin/noisy_eval_20260913/results/report.md) links all
22 cases, and `results.json`/`results.csv` contain every score, failure reason
and input/artifact hash. Files are ignored generated artifacts; running the
commands above recreates them. Useful individual listening comparisons:

- Café, 2.5 dB: [noisy](dpdfnet_bin/noisy_eval_20260913/results/p257_009/noisy.wav),
  [FPGA](dpdfnet_bin/noisy_eval_20260913/results/p257_009/fpga.wav),
  [clean](dpdfnet_bin/noisy_eval_20260913/results/p257_009/clean.wav).
- Accuracy failure: [noisy](dpdfnet_bin/noisy_eval_20260913/results/p257_018/noisy.wav),
  [FPGA](dpdfnet_bin/noisy_eval_20260913/results/p257_018/fpga.wav),
  [CPU](dpdfnet_bin/noisy_eval_20260913/results/p257_018/cpu.wav).
- User sample: [FPGA WAV](dpdfnet_bin/p232_007_enhanced_fpga.wav),
  [CPU WAV](dpdfnet_bin/p232_007_enhanced_cpu.wav).

All 63 DPDFNet host tests pass, including metric edge cases, paired CPU/FPGA
comparison and aggregation with missing scores/clean controls. Source download,
cache reuse and corruption rejection checks also pass. This is a small, balanced functional evaluation;
it does not establish full-corpus quality, live-room robustness or a human
listening score. The identified onset error still needs investigation.
