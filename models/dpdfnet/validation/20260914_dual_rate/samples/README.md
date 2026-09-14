# Matched noisy, CPU and FPGA listening samples

Each row contains the same complete recording at the stated model rate.
Files are mono floating-point WAVs with no normalization, gain fitting or
clipping. CPU and FPGA outputs use the same fixed model-delay compensation.
The original recordings are 48 kHz; these copies are resampled to the model's
native rate. The 8-kHz copies therefore have lower audio bandwidth.

| Model rate | Recording | Noisy | CPU | FPGA |
| --- | --- | --- | --- | --- |
| 8 kHz | `p232_007`, café, 3.956 s | [WAV](8k/p232_007_noisy.wav) | [WAV](8k/p232_007_cpu.wav) | [WAV](8k/p232_007_fpga.wav) |
| 8 kHz | `p257_018`, public square, 4.419 s | [WAV](8k/p257_018_noisy.wav) | [WAV](8k/p257_018_cpu.wav) | [WAV](8k/p257_018_fpga.wav) |
| 16 kHz | `p232_007`, café, 3.956 s | [WAV](16k/p232_007_noisy.wav) | [WAV](16k/p232_007_cpu.wav) | [WAV](16k/p232_007_fpga.wav) |
| 16 kHz | `p257_018`, public square, 4.419 s | [WAV](16k/p257_018_noisy.wav) | [WAV](16k/p257_018_cpu.wav) | [WAV](16k/p257_018_fpga.wav) |

Every enhanced file is reconstructed from the saved streaming spectrum output
of its corresponding pinned CPU model or resident FPGA bin. Source-rate
waveform error, timing and limitations are in the
[benchmark report](../../../DUAL_RATE_BENCHMARK.md). The existing 16-kHz bin's
onset accuracy problem remains unresolved. These listening samples do not
establish that every model output improves speech quality.

File hashes, source hashes and exact sample counts are in
[listening_samples.json](../listening_samples.json). The native 8-kHz sample
counts are 31,648 and 35,349; 16-kHz counts are 63,295 and 70,698.

Source attribution: Cassia Valentini-Botinhao (2017), University of Edinburgh /
CSTR, *Noisy speech database for training speech enhancement algorithms and
TTS models, 2016*, [DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117),
CC BY 4.0. Environmental recordings originate from [DEMAND](https://zenodo.org/records/1227121),
Joachim Thiemann, Nobutaka Ito and Emmanuel Vincent (2013), CC BY-SA 3.0.
These files are adaptations through resampling and, for CPU/FPGA outputs,
speech enhancement. Original archive provenance is retained in the repository's
[noisy-test documentation](../../../NOISY_TESTS.md) and the benchmark source
manifest. No new speech or noise was synthesized.
