# BigCodec: eight noisy audio files on Italy

Eight mono 16 kHz files, **173.010 seconds** total; each file runs as one complete encoder → quantizer → decoder graph. BigCodec reconstructs compressed speech; these tests measure codec implementation accuracy, not denoising.

RK, Kintex UltraScale+ KU5P, AXI256, hardware `0x40519e0a`, **333.25 MHz**, 2 GiB visible DRAM. CPU reference: official FP32 model, Intel Core Ultra 9 285K, one thread on core 7; FPGA host on core 6.

**RTF = processing time / audio duration; lower is faster and real time requires ≤1.** Timings include audio preprocessing, transfers, execution and output writes; they exclude compilation, model/artifact loading and resident-bin upload. Each file has one measured run per variant.

Runner total RTF uses the separately measured `total_elapsed_s`, including artifact loading, validation and resident upload before audio processing. It is not reconstructed by adding only the load/upload sub-timers; Python startup and final report serialization fall outside that runner timer.

| Implementation | Processing (s) | Processing RTF | Runner total RTF | Waveform error vs CPU | Matching CPU tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU FP32 | 402.021 | 2.32369 | — | reference | 13845/13845 |
| FPGA BF16 | 829.847 | 4.79653 | 6.56707 | 28.06% | 12607/13845 |
| FPGA IF8 recurrence | 753.544 | 4.35550 | 6.04021 | 27.04% | 12617/13845 |

Both FPGA variants use BF16 convolutions and activations. “IF8 recurrence” changes the recurrent weights in both encoder and decoder LSTMs to IF8-INT; LSTM input weights remain BF16.

WAV links below preserve the complete input sample counts. “Error” is `100 × ||FPGA − CPU||₂ / ||CPU||₂`, with no gain, delay or polarity fitting. Pooling sums squared error and CPU energy before division; it does not average per-file percentages.

| Input WAV | Duration (s) | CPU WAV | BF16 WAV / RTF | BF16 error | IF8 recurrence WAV / RTF | IF8 error |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| [Bus](../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav) | 23.682 | [CPU](cpu/bus.wav) | [4.79644](fpga_bf16/bus.wav) | 28.09% | [4.35501](fpga_if8/bus.wav) | 27.45% |
| [Café](../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_noisy.wav) | 20.509 | [CPU](cpu/cafe.wav) | [4.79472](fpga_bf16/cafe.wav) | 21.96% | [4.35396](fpga_if8/cafe.wav) | 24.75% |
| [Office](../../../dpdfnet/validation/20260914_noisy20s/noisy/office_noisy.wav) | 21.980 | [CPU](cpu/office.wav) | [4.79645](fpga_bf16/office.wav) | 23.84% | [4.35558](fpga_if8/office.wav) | 22.18% |
| [Public square](../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_noisy.wav) | 21.473 | [CPU](cpu/psquare.wav) | [4.79566](fpga_bf16/psquare.wav) | 33.45% | [4.35493](fpga_if8/psquare.wav) | 34.86% |
| [Bus · 2.5 dB](../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) | 20.879 | [CPU](cpu/bus_low_snr.wav) | [4.79737](fpga_bf16/bus_low_snr.wav) | 22.80% | [4.35616](fpga_if8/bus_low_snr.wav) | 24.14% |
| [Café · 2.5 dB](../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) | 21.466 | [CPU](cpu/cafe_low_snr.wav) | [4.79755](fpga_bf16/cafe_low_snr.wav) | 36.99% | [4.35639](fpga_if8/cafe_low_snr.wav) | 35.99% |
| [Office · 2.5 dB](../../../dpdfnet/validation/20260914_noisy20s/noisy/office_low_snr_noisy.wav) | 21.494 | [CPU](cpu/office_low_snr.wav) | [4.79653](fpga_bf16/office_low_snr.wav) | 19.64% | [4.35556](fpga_if8/office_low_snr.wav) | 19.88% |
| [Public square · 2.5 dB](../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_low_snr_noisy.wav) | 21.525 | [CPU](cpu/psquare_low_snr.wav) | [4.79748](fpga_bf16/psquare_low_snr.wav) | 32.18% | [4.35639](fpga_if8/psquare_low_snr.wav) | 24.16% |

The four mixed-SNR files cover the original 2.5, 7.5, 12.5 and 17.5 dB conditions. The four 2.5 dB files use distinct additional utterances selected before evaluation. All files concatenate complete VoiceBank-DEMAND mixtures with 250 ms zero gaps; see [attribution and construction](ATTRIBUTION.md).

All 16 FPGA outputs are finite, have zero padding lanes and preserve source lengths. Every run records **one resident program/parameter upload, one whole-file input upload, one START, one HALT and one waveform/token bundle read**; no CPU neural operations. This is whole-utterance execution, with recurrent state retained across the file and reset between files. A token spans 200 samples (12.5 ms), not an independently processed 10 ms block.

The waveform and token differences remain material implementation limitations. These measurements do not establish equivalence to the official FP32 decoder, perceptual quality, or real-time operation.

| Resident bin component | BF16 range (bytes) | IF8 recurrence range (bytes) |
| --- | ---: | ---: |
| Parameters/constants | 331,251,584 | 294,682,496 |
| Instructions | 432,926,784–500,013,056 | 415,282,752–479,638,016 |
| Complete resident image | 764,178,368–831,264,640 | 709,965,248–774,320,512 |

Each bin is compiled for its complete padded file length. Long-file captures use model/program addresses `0x90000000–0xD0000000` and tensors/workspace `0xD0000000–0x100000000`; input starts at `0x80000000`. Legacy short-bin addresses remain supported. Exact instruction counts, parameter/program bytes and bin SHA-256 values for all 16 runs are in [results.json](results.json) and [hardware run records](hardware_runs.json).

From the repository root, with the BigCodec Python environment active, compile and run the complete bus file. These example paths are separate from the measured artifacts:

```bash
python models/bigcodec/bigcodec_compile.py \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav \
  --conv-precision bf16 --lstm-precision bf16 \
  --output models/bigcodec/bigcodec_bin/bus-example-bf16.bin
python models/bigcodec/bigcodec_run_from_bin.py \
  --bin models/bigcodec/bigcodec_bin/bus-example-bf16.bin \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav \
  --output /tmp/bigcodec-bus-example.wav
```

[Download all 32 WAVs](bigcodec_noisy_audio_wavs.zip): eight unchanged inputs, eight CPU reconstructions and sixteen FPGA reconstructions. [Audio hashes](audio_manifest.json) identify every file; [published artifacts](published_artifacts.json) records the ZIP hash.

[Independent audit](independent_audit.json) · [107 software tests](software_tests.log) · [upper-DDR hardware test](high_dram_smoke.json) · [legacy-bin equivalence](legacy_layout_hash_verification.json) · [validation provenance](validation_provenance.json).

[Summary](summary.json) · [CPU provenance](cpu_reference_manifest.json) · [hardware source hashes](hardware_source_hashes.json). The initial bus/BF16 pilot preceded the capacity preflight; its runner hash override is recorded per run, and the board’s 2 GiB upper-address test passed independently.
