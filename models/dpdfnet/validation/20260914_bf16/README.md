# DPDFNet BF16 benchmark

**8-kHz BF16 RTF: 1.044; 16-kHz BF16 RTF: 1.499.** Both exceed the real-time threshold of 1 for the complete audio pipeline.

Measured on Italy, Kintex UltraScale+ KU5P, RK AXI-256, build `40519e0a`, 333.25 MHz. Host: Intel Core Ultra 9 285K (rated maximum turbo 5.7 GHz), one math thread, CPU core 6. Both precisions were rerun on this build using the same bus and café recordings at 2.5 dB source SNR: 20.879 s and 21.466 s, respectively. Each row covers 42.345 s and 4,247 frames.

| Model | Dense weights | FPGA ms/frame | Host ms/frame | Audio RTF | Real-time speed |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 kHz | BF16 | 9.591 | 10.398 | **1.044** | 0.957× |
| 8 kHz | IF8 | 9.070 | 9.865 | **0.991** | 1.009× |
| 16 kHz | BF16 | 14.085 | 14.936 | **1.499** | 0.667× |
| 16 kHz | IF4/IF8 | 13.447 | 14.291 | **1.434** | 0.697× |

RTF = processing time / recording duration; lower is faster. Audio RTF includes host audio preprocessing, every inference frame, reconstruction and WAV writing. It excludes one-time bin loading, reset and program/parameter upload. No warmup frames are discarded. Input and output WAVs retain the original 16-kHz sample rate and exact sample count; the 8-kHz model resamples internally.

8-kHz BF16 finishes FPGA execution within 10 ms (maximum 9.601 ms), but host transfers and processing raise average frame time to 10.398 ms. Compared with the quantized controls, BF16 increases audio processing time by 5.39% at 8 kHz and 4.51% at 16 kHz.

| Model / dense weights | Host frames >10 ms | FPGA frames >10 ms | Waveform error vs FP32 CPU |
| --- | ---: | ---: | ---: |
| 8 kHz / BF16 | 4,100/4,247 | 0/4,247 | 2.728% |
| 8 kHz / IF8 | 838/4,247 | 0/4,247 | 2.675% |
| 16 kHz / BF16 | 4,247/4,247 | 4,247/4,247 | 3.189% |
| 16 kHz / IF4/IF8 | 4,247/4,247 | 4,247/4,247 | 5.035% |

All output spectra and WAV samples are finite. Waveform error is pooled relative L2 against each model’s SHA256-verified ONNX FP32 CPU output, using every sample without fitted gain, alignment or trimming. It measures numerical agreement, not clean-reference speech quality. CPU references are reused from the earlier noisy-audio validation. All four quantized control outputs match the earlier `df0749de` results sample for sample. [Independent results audit](independent_audit.json).

Both BF16 bins contain the full neural graph, parameters and initial state in one image. That image is uploaded once. Each 10-ms hop uploads one BF16 spectrum, issues one START, observes one HALT and reads one enhanced spectrum; recurrent state stays in DRAM. Host STFT/iSTFT converts between audio and spectra. [Observed transfer/START/HALT audit](frame_contract.json).

BF16 dense convolution uses device im2col and BF16 matrix instructions. The quantized controls use native CONV instructions, so this compares complete software implementations as well as weight precision. BF16 describes stored weights and activations; the existing internal BF19/BF20 arithmetic is unchanged.

## Audio samples

| Input | BF16 8-kHz output | BF16 16-kHz output |
| --- | --- | --- |
| [bus_low_snr_noisy.wav](../20260914_noisy20s/noisy/bus_low_snr_noisy.wav) | [bus_low_snr_fpga_8khz_bf16.wav](fpga8k_bf16/bus_low_snr_fpga_8khz_bf16.wav) (RTF 1.045) | [bus_low_snr_fpga_16khz_bf16.wav](fpga16k_bf16/bus_low_snr_fpga_16khz_bf16.wav) (RTF 1.499) |
| [cafe_low_snr_noisy.wav](../20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) | [cafe_low_snr_fpga_8khz_bf16.wav](fpga8k_bf16/cafe_low_snr_fpga_8khz_bf16.wav) (RTF 1.044) | [cafe_low_snr_fpga_16khz_bf16.wav](fpga16k_bf16/cafe_low_snr_fpga_16khz_bf16.wav) (RTF 1.499) |

| Input case | Quantized 8-kHz output | Quantized 16-kHz output |
| --- | --- | --- |
| bus_low_snr | [bus_low_snr_fpga_8khz_quantized.wav](fpga8k_quantized/bus_low_snr_fpga_8khz_quantized.wav) | [bus_low_snr_fpga_16khz_quantized.wav](fpga16k_quantized/bus_low_snr_fpga_16khz_quantized.wav) |
| cafe_low_snr | [cafe_low_snr_fpga_8khz_quantized.wav](fpga8k_quantized/cafe_low_snr_fpga_8khz_quantized.wav) | [cafe_low_snr_fpga_16khz_quantized.wav](fpga16k_quantized/cafe_low_snr_fpga_16khz_quantized.wav) |

Source attribution and construction: [noisy-audio report](../20260914_noisy20s/README.md). These are the same recordings and contain no newly mixed noise.

## BF16 bins

| Model | Instructions | Program bytes | Parameter/data bytes | Resident bytes |
| --- | ---: | ---: | ---: | ---: |
| 8 kHz BF16 | 28,058 | 897,856 | 14,538,368 | 15,436,224 |
| 16 kHz BF16 | 41,090 | 1,314,880 | 13,220,224 | 14,535,104 |

Parameter/data bytes include constants, initial state and padding. The program and data sections are bundled in one deployment bin. Offline checks passed: 174 software tests; all 15 dense convolution weight/bias byte blocks per model match rounded ONNX BF16; no quantized compute descriptors; one terminal HALT and no software interrupts. Default deployments remain byte-identical. [8-kHz precision audit](precision8k.json), [16-kHz precision audit](precision16k.json), [default-bin regression](default_bin_regression.json).

Build from the repository root (add `--force` to rebuild an existing BF16 bin):

```bash
python models/dpdfnet8khz/dpdfnet8khz_compile.py --conv-precision bf16
python models/dpdfnet/dpdfnet_compile.py --conv-precision bf16
```

Named test entry points select and check the dense convolution weight precision:

| Model | BF16 test | Quantized test |
| --- | --- | --- |
| 8 kHz | [dpdfnet8khz_bf16_weights_test.py](../../../dpdfnet8khz/dpdfnet8khz_bf16_weights_test.py) | [dpdfnet8khz_if8_weights_test.py](../../../dpdfnet8khz/dpdfnet8khz_if8_weights_test.py) |
| 16 kHz | [dpdfnet16khz_bf16_weights_test.py](../../dpdfnet16khz_bf16_weights_test.py) | [dpdfnet16khz_if4_if8_weights_test.py](../../dpdfnet16khz_if4_if8_weights_test.py) |

Run 8-kHz BF16 on Italy, reusing the existing CI lock with read-only access:

```bash
(
  flock -n 9 || exit 1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/my_torch_env/bin/python \
    models/dpdfnet8khz/dpdfnet8khz_bf16_weights_test.py \
    --device rk --dev xdma0 --cpu-core 6 \
    --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav \
    --output /tmp/bus_low_snr_fpga_8khz_bf16.wav
) 9</tmp/pcie_ci_hw_italy.lock
```

For 16 kHz, use `models/dpdfnet/dpdfnet16khz_bf16_weights_test.py`. Read `steady_audio_rtf` in the output’s `.metrics.json` file. The named tests use the same measured inference implementation and distinct default output filenames; mismatched `--bin` precision is rejected before FPGA access.

[Summary](summary.json) · [Per-run commands, hashes and results](results.json). Raw logs and every frame’s timing are saved beside each output WAV.
