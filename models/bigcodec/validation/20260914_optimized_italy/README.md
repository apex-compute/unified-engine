# BigCodec optimized FPGA benchmarks

**The RTF 4–5 target is met on this recording.** Optimized BF16 takes **19.146 s / RTF 4.840**, versus **60.615 s / RTF 15.323** for the rerun baseline. Every reconstructed waveform sample and all 317 token IDs are bit-identical to the baseline: **3.166× faster with unchanged output**. BF16 is the prepared default bin on Italy.

Measured on Italy, Kintex UltraScale+ KU5P, RK AXI256, build `0x40519e0a`, hardware-info `0x80214d40`, reported 333.25 MHz. Host: Intel Core Ultra 9 285K, FPGA runner on core 6 and official FP32 reference on core 7, one math thread. The baseline also exactly reproduces the [previous FPGA build's output](../20260914_italy/README.md).

## Performance and accuracy

Input: **3.955896 s** of VoiceBank-DEMAND noisy speech, café noise at 12.5 dB nominal SNR. The 48 kHz file has 189,883 samples; BigCodec resamples it to 63,295 samples at its native 16 kHz and pads to 63,400 samples / 317 tokens. [Input WAV](../../../../test_samples/p232_007.wav), [source and license](../../../../test_samples/p232_007.README.md).

| Implementation | Processing | RTF | Speedup over FPGA baseline | CPU token matches | Waveform error vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official CPU FP32 | 6.486 s | 1.639 | — | 317/317 | Reference |
| FPGA BF16 baseline | 60.615 s | 15.323 | 1.000× | 301/317 | 19.287% |
| Optimized BF16 | 19.146 s | **4.840** | **3.166×** | 301/317 | 19.287% |
| Optimized, encoder IF8 recurrence | 18.275 s | **4.620** | **3.317×** | 303/317 | 21.349% |
| Optimized, both IF8 recurrent stacks | 17.411 s | **4.401** | **3.481×** | 303/317 | 17.446% |

All FPGA rows use BF16 convolutions, input projections and activation/state arithmetic. IF8 applies only to the named recurrent weight matrices. Each row is one full-recording run on the same FPGA build. These are measurements for this input, not a latency guarantee for other utterances.

RTF = processing time / audio duration; real time requires RTF ≤ 1. Processing includes audio preprocessing, input/output DMA, device execution and output restoration/write. It excludes compilation, bin/checkpoint loading and the one-time model upload. Native FPGA execution is 18.976 s for optimized BF16 and 17.229 s for both-stack IF8. The runner's total elapsed time including initialization/loading is 26.678 s / 24.870 s respectively; model upload is 0.144 s / 0.132 s. The fresh CPU reference exactly reproduces the earlier FP32 waveform and tokens.

Waveform error is `||FPGA − CPU||₂ / ||CPU||₂`, with original sample indices and no gain, delay or polarity fitting. Decoding the same FPGA token IDs on CPU isolates decoder differences:

| Variant | Error vs FP32 decoder using the same token IDs |
| --- | ---: |
| Baseline and optimized BF16 | 34.954% |
| Encoder IF8 recurrence | 31.269% |
| Both IF8 recurrent stacks | 45.472% |

These errors use different reference waveforms from the first table and do not add. Both-stack IF8 improves the end-to-end error on this clip while worsening the isolated decoder error. The existing difference from FP32 remains a limitation of this experimental port.

## Audio and run records

| Output | Run record | CPU comparison |
| --- | --- | --- |
| [CPU reference](p232_007_cpu.wav) | [metrics](p232_007_cpu.metrics.json) | Reference |
| [FPGA baseline](p232_007_baseline_fpga.wav) | [metrics](p232_007_baseline_fpga.metrics.json) | [comparison](p232_007_baseline_fpga_comparison.json) |
| [Optimized BF16](p232_007_optimized_bf16_fpga.wav) | [metrics](p232_007_optimized_bf16_fpga.metrics.json) | [comparison](p232_007_optimized_bf16_fpga_comparison.json) |
| [Encoder IF8](p232_007_optimized_encoder_if8_fpga.wav) | [metrics](p232_007_optimized_encoder_if8_fpga.metrics.json) | [comparison](p232_007_optimized_encoder_if8_fpga_comparison.json) |
| [Both-stack IF8](p232_007_optimized_lstm_if8_fpga.wav) | [metrics](p232_007_optimized_lstm_if8_fpga.metrics.json) | [comparison](p232_007_optimized_lstm_if8_fpga_comparison.json) |

Token NPZ files and same-token CPU decoder WAVs accompany these records. [Summary](summary.json) and [artifact hashes](artifact_manifest.json) bind the results to the tested binaries and source. Raw records retain the paths used during execution; collected copies are identified by their hashes.

## Bins and execution

| Variant | Parameters/constants | Instruction program | Resident image | Instructions |
| --- | ---: | ---: | ---: | ---: |
| BF16 baseline | 324.166 MB | 165.541 MB | 489.706 MB | 5,173,150 |
| Optimized BF16 | 331.252 MB | 83.999 MB | 415.251 MB | 2,624,968 |
| Encoder IF8 recurrence | 312.967 MB | 82.295 MB | 395.262 MB | 2,571,712 |
| Both IF8 recurrent stacks | 294.682 MB | 80.591 MB | 375.273 MB | 2,518,456 |

MB means 1,000,000 bytes. [Compile records](compiled_artifacts.json) contain exact sizes, local bin paths and SHA256 hashes. Generated bins stay outside Git.

Every FPGA run uploads one resident program/parameter image, uploads the padded waveform to DRAM, issues **one START**, observes **one HALT**, and reads one waveform/token bundle. Host neural operations: **0**. All outputs are finite and all unused waveform lanes are zero. This is whole-utterance inference with a 12.5 ms token hop; it does not implement independent 10 ms streaming.

## Implementation checks

Convolutions reuse SRAM windows and weight strips; FIR/Snake, recurrent state updates and early codebook tournament rounds retain intermediate values in SRAM. Hardware validation also fixed two layout faults: SRAM copy now multiplies by exact one, and final Tanh transfers complete 128-byte sample rows. Final waveform parity proves these fixes preserve the BF16 baseline on this recording.

All **97 software tests pass** ([output](software_tests.txt)). Hardware records cover [convolutions](convolution_checks.json), [copies/clamps, activations and codebook selection](native_sram_checks.json), [BF16/IF8 recurrence](lstm_checks.json), and [audio DMA/Tanh layout](waveform_layout_checks.json). The layout probes intentionally include unsupported 32/64-byte scatter controls; corrected 128-byte transfers and final Tanh pass exactly. On this build, MAXPOOL and multiply-one flush magnitudes below 2⁻¹²⁷ to signed zero; the positive mask treats those positive gaps as ties. Normal values pass exactly; these checks do not assert full IEEE bit equivalence.

The corresponding hardware scripts are included here and require `--execute`. Run them serially with `--expected-version 0x40519e0a`; they acquire Italy's shared hardware lock before accessing the device.

## Run the measured BF16 model

From the repository root, with dependencies installed:

```bash
python models/bigcodec/bigcodec_compile.py \
  --input test_samples/p232_007.wav --conv-precision bf16 --lstm-precision bf16 \
  --output /tmp/bigcodec-optimized.bin

python models/bigcodec/bigcodec_run_from_bin.py \
  --bin /tmp/bigcodec-optimized.bin --input test_samples/p232_007.wav \
  --output /tmp/bigcodec-optimized.wav
```

On Italy the default `models/bigcodec/bigcodec_bin/bigcodec-andromeda.bin` already contains this validated BF16 model, so the runner can omit `--bin`. Use `--lstm-precision encoder-if8` or `--lstm-precision if8` with a separate compiled bin for the other measured variants.
