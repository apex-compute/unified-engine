# DPDFNet: four-variant RK-256 performance report

**8-kHz IF8 reaches average real-time throughput (RTF 0.991). 8-kHz BF16 is slightly slower than real time (1.044); both 16-kHz variants exceed 1.4. Every variant has host frames that miss the 10-ms deadline.**

Measurements recorded on **2026-09-14**, on Italy's **Kintex UltraScale+ KU5P**, **RK AXI-256**, FPGA build **`40519e0a`**. The measured `HW_INFO` value is `0x80214d40`, reporting **333.25 MHz**; no clock override or tracing was enabled. Host: **Intel Core Ultra 9 285K**, rated maximum turbo **5.7 GHz**, with the FPGA runner pinned to core 6 and one OMP/MKL thread. The turbo rating is not a measured execution clock.

All four variants processed the same bus and café recordings, **20.879 s** and **21.466 s**: **42.345 s and 4,247 frames per variant**, including the trailing frames needed to recover the end of each recording. This report consolidates the [saved measurements](validation/20260914_bf16/results.json) from that build.

## Throughput

| Model / dense weights | FPGA mean ms/frame | Host mean ms/frame | Audio processing s | Audio RTF | Real-time speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 kHz / BF16 | 9.591 | 10.398 | 44.228 | **1.044** | 0.957× |
| 8 kHz / IF8 | 9.070 | 9.865 | 41.965 | **0.991** | 1.009× |
| 16 kHz / BF16 | 14.085 | 14.936 | 63.481 | **1.499** | 0.667× |
| 16 kHz / IF4/IF8 | 13.447 | 14.291 | 60.740 | **1.434** | 0.697× |

**RTF = processing time / audio duration; speed = 1 / RTF.** Audio processing includes host preprocessing, all inference frames, audio reconstruction and WAV writing. It excludes one-time bin loading, reset and program/parameter upload. No warmup frames are discarded. Host frame timing includes input packing and transfer, START/wait/HALT and output read; STFT/iSTFT and file I/O are counted in audio RTF separately.

BF16 increases audio processing time by **5.39% at 8 kHz** and **4.51% at 16 kHz** compared with the corresponding quantized implementation. Both 8-kHz variants complete every measured FPGA execution within 10 ms: maxima are **9.601 ms BF16** and **9.077 ms IF8**. Host overhead makes 8-kHz BF16 miss average real time. Both 16-kHz variants already exceed the deadline during FPGA execution.

## Deadlines and agreement with CPU

| Model / dense weights | Host p95 ms | Host p99 ms | Host maximum ms | Host frames >10 ms | FPGA frames >10 ms | Waveform error vs FP32 CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 kHz / BF16 | 10.692 | 10.776 | 15.302 | 4,100/4,247 | 0/4,247 | 2.728% |
| 8 kHz / IF8 | 10.134 | 10.602 | 13.908 | 838/4,247 | 0/4,247 | 2.675% |
| 16 kHz / BF16 | 15.209 | 15.339 | 19.459 | 4,247/4,247 | 4,247/4,247 | 3.189% |
| 16 kHz / IF4/IF8 | 14.576 | 14.705 | 17.855 | 4,247/4,247 | 4,247/4,247 | 5.035% |

8-kHz IF8 meets average throughput with little margin; **838 of 4,247 host frames exceed 10 ms**. These measurements do not establish that every hop can meet its deadline. Each model also has a separate **40-ms algorithmic delay**.

All output spectra and WAV samples were finite. Waveform error is pooled relative L2, `sqrt(sum((FPGA - CPU)^2) / sum(CPU^2))`, over every original source-rate sample, without fitted gain, alignment or trimming. The references are SHA256-verified outputs from the corresponding pinned ONNX FP32 model, run with one ONNX Runtime thread on core 7. Those references were reused; CPU performance was not remeasured for this comparison. The error measures numerical agreement, not clean-reference speech quality. BF16 improves 16-kHz agreement, while 8-kHz agreement is slightly worse on these two recordings.

Precision names describe **dense convolution weights**: all 15 kernels use BF16 or IF8 at 8 kHz; the 16-kHz quantized implementation uses **12 IF4 and 3 IF8 kernels**. Other learned weights and activations use BF16. BF16 dense convolutions use device im2col and BF16 matrix instructions; quantized convolutions use native CONV instructions. This compares the complete implementations as well as weight precision. Internal BF19/BF20 arithmetic is unchanged.

## Single-bin execution and sizes

Each variant uses **one deployment bin containing the complete neural program, parameters and initial state**, uploaded once to DRAM at `0x90000000`. For every 10-ms hop, the host uploads one padded BF16 spectrum, issues **one START**, waits for **one HALT**, and reads one enhanced spectrum. Recurrent state remains in device DRAM. Host STFT/iSTFT converts between PCM audio and spectra; there are no host neural operations or intermediate tensor transfers.

| Native model | New audio samples per hop | ONNX graph nodes | Input bytes per hop | Output bytes per hop |
| --- | ---: | ---: | ---: | ---: |
| 8 kHz | 80 | 492 | 10,368 | 10,368 |
| 16 kHz | 160 | 472 | 20,608 | 20,608 |

All eight runs record one model upload and one input write, START, HALT and output read per frame. A separate [observed transfer audit](validation/20260914_bf16/frame_contract.json) verifies the driver calls for two frames of each BF16 model.

| Model / dense weights | Instructions | Program bytes | Parameter/data bytes | Resident upload bytes | Serialized .bin bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 kHz / BF16 | 28,058 | 897,856 | 14,538,368 | 15,436,224 | 15,566,040 |
| 8 kHz / IF8 | 27,118 | 867,776 | 22,726,272 | 23,594,048 | 23,721,269 |
| 16 kHz / BF16 | 41,090 | 1,314,880 | 13,220,224 | 14,535,104 | 14,658,165 |
| 16 kHz / IF4/IF8 | 40,264 | 1,288,448 | 16,575,360 | 17,863,808 | 17,984,019 |

Program and parameter/data bytes are sections of the same bin. Parameter/data includes constants, initial state, padding and replicated weight streams; resident upload size excludes the serialization container and runtime tensor workspace. These sizes reflect each compiler's layout and stored constants, not just learned-weight precision. Serialized sizes describe the exact measured files.

| Model / dense weights | Deployment filename | Test entry point |
| --- | --- | --- |
| 8 kHz / BF16 | `dpdfnet2_8khz-bf16-andromeda.bin` | [dpdfnet8khz_bf16_weights_test.py](../dpdfnet8khz/dpdfnet8khz_bf16_weights_test.py) |
| 8 kHz / IF8 | `dpdfnet2_8khz-andromeda.bin` | [dpdfnet8khz_if8_weights_test.py](../dpdfnet8khz/dpdfnet8khz_if8_weights_test.py) |
| 16 kHz / BF16 | `dpdfnet2-bf16-andromeda.bin` | [dpdfnet16khz_bf16_weights_test.py](dpdfnet16khz_bf16_weights_test.py) |
| 16 kHz / IF4/IF8 | `dpdfnet2-andromeda.bin` | [dpdfnet16khz_if4_if8_weights_test.py](dpdfnet16khz_if4_if8_weights_test.py) |

Bins are generated under `models/dpdfnet8khz/dpdfnet8khz_bin/` or `models/dpdfnet/dpdfnet_bin/`. Each test selects its matching bin and checks encoded precision before FPGA access. Recorded bin SHA256 values are in [results.json](validation/20260914_bf16/results.json). Current default compiles preserve the measured program and resident data exactly; descriptive metadata and serialized file hashes can differ, as recorded in the [default-bin regression](validation/20260914_bf16/default_bin_regression.json).

## Noisy audio results

Both inputs concatenate existing VoiceBank/DEMAND clips with nominal **2.5-dB source SNR**, separated by 250-ms silence. No additional noise was mixed in. [Source attribution and construction](validation/20260914_noisy20s/README.md).

| Input file | Duration | Source samples | Inference frames |
| --- | ---: | ---: | ---: |
| [bus_low_snr_noisy.wav](validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) | 20.879 s | 334,067 | 2,094 |
| [cafe_low_snr_noisy.wav](validation/20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) | 21.466 s | 343,450 | 2,153 |

| Model / dense weights | FPGA output file | Audio RTF | Waveform error vs FP32 CPU |
| --- | --- | ---: | ---: |
| 8 kHz / BF16 | [bus_low_snr_fpga_8khz_bf16.wav](validation/20260914_bf16/fpga8k_bf16/bus_low_snr_fpga_8khz_bf16.wav) | 1.045 | 1.570% |
| 8 kHz / BF16 | [cafe_low_snr_fpga_8khz_bf16.wav](validation/20260914_bf16/fpga8k_bf16/cafe_low_snr_fpga_8khz_bf16.wav) | 1.044 | 3.319% |
| 8 kHz / IF8 | [bus_low_snr_fpga_8khz_quantized.wav](validation/20260914_bf16/fpga8k_quantized/bus_low_snr_fpga_8khz_quantized.wav) | 0.992 | 1.593% |
| 8 kHz / IF8 | [cafe_low_snr_fpga_8khz_quantized.wav](validation/20260914_bf16/fpga8k_quantized/cafe_low_snr_fpga_8khz_quantized.wav) | 0.990 | 3.236% |
| 16 kHz / BF16 | [bus_low_snr_fpga_16khz_bf16.wav](validation/20260914_bf16/fpga16k_bf16/bus_low_snr_fpga_16khz_bf16.wav) | 1.499 | 1.397% |
| 16 kHz / BF16 | [cafe_low_snr_fpga_16khz_bf16.wav](validation/20260914_bf16/fpga16k_bf16/cafe_low_snr_fpga_16khz_bf16.wav) | 1.499 | 4.012% |
| 16 kHz / IF4/IF8 | [bus_low_snr_fpga_16khz_quantized.wav](validation/20260914_bf16/fpga16k_quantized/bus_low_snr_fpga_16khz_quantized.wav) | 1.434 | 2.726% |
| 16 kHz / IF4/IF8 | [cafe_low_snr_fpga_16khz_quantized.wav](validation/20260914_bf16/fpga16k_quantized/cafe_low_snr_fpga_16khz_quantized.wav) | 1.435 | 6.188% |

Every output is a mono floating-point WAV at the original **16-kHz file sample rate**, with exactly the original sample count. The 8-kHz model resamples internally and restores the file rate after inference. Output links use the actual recorded filenames; `quantized` identifies IF8 at 8 kHz and IF4/IF8 at 16 kHz. Each output has a sibling `.metrics.json` and `.log` with full frame timings and hashes.

## Reproduce

From the repository root, use the Python environment with the repository's matching `torch` and `torchaudio` packages. On Italy this is `~/my_torch_env/bin/python`.

```bash
~/my_torch_env/bin/python -m pip install -r models/dpdfnet8khz/requirements.txt

~/my_torch_env/bin/python models/dpdfnet8khz/dpdfnet8khz_compile.py --download --conv-precision bf16
~/my_torch_env/bin/python models/dpdfnet8khz/dpdfnet8khz_compile.py --download --conv-precision if8
~/my_torch_env/bin/python models/dpdfnet/dpdfnet_compile.py --download --conv-precision bf16
~/my_torch_env/bin/python models/dpdfnet/dpdfnet_compile.py --download --conv-precision auto
```

Compilation does not access the FPGA. Skip it when the matching bins already exist, or add `--force` to rebuild. Run each variant on the bus recording with the following commands; substitute `cafe_low_snr_noisy.wav` to test the café recording. The existing CI lock is opened read-only, acquired without queuing for each run, and released when that run ends.

```bash
run_dpdfnet() (
  flock -n 9 || exit 1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/my_torch_env/bin/python "$1" \
    --device rk --dev xdma0 --cpu-core 6 --timeout 10 \
    --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav \
    --output "$2"
) 9</tmp/pcie_ci_hw_italy.lock

run_dpdfnet models/dpdfnet8khz/dpdfnet8khz_bf16_weights_test.py /tmp/bus_low_snr_fpga_8khz_bf16.wav
run_dpdfnet models/dpdfnet8khz/dpdfnet8khz_if8_weights_test.py /tmp/bus_low_snr_fpga_8khz_if8.wav
run_dpdfnet models/dpdfnet/dpdfnet16khz_bf16_weights_test.py /tmp/bus_low_snr_fpga_16khz_bf16.wav
run_dpdfnet models/dpdfnet/dpdfnet16khz_if4_if8_weights_test.py /tmp/bus_low_snr_fpga_16khz_if4_if8.wav
```

Read `steady_audio_rtf` and `host_frame_deadline_misses` in each output's `.metrics.json`. Use a compatible RK AXI-256 queue-CONFIG/CONV FPGA image; `--device rk` does not select or flash a bitstream. The saved [run commands](validation/20260914_bf16/results.json) used the shared runners with explicit bins; the named tests call those same inference implementations and add precision validation.

## Validation evidence

The [independent audit](validation/20260914_bf16/independent_audit.json) recomputes timing, deadline counts and waveform errors, verifies hashes and transfer counts for all eight runs, and confirms that the four quantized output WAVs match the earlier build's outputs sample for sample. The [8-kHz](validation/20260914_bf16/precision8k.json) and [16-kHz](validation/20260914_bf16/precision16k.json) BF16 audits verify all 15 dense weight/bias blocks against rounded ONNX values, the bound DMA/matrix instructions, one terminal HALT and no software interrupts. [174 software tests passed](validation/20260914_bf16/software_tests.json).

[Aggregate measurements](validation/20260914_bf16/summary.json) · [Per-run results](validation/20260914_bf16/results.json) · [Evidence checksums](validation/20260914_bf16/SHA256SUMS)
