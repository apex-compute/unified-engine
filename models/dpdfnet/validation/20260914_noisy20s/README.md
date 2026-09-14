# DPDFNet2 FPGA noisy-audio results

Eight tests cover bus, café, office and public-square noise, with noisy
inputs and actual FPGA outputs from the 8-kHz and 16-kHz models.

## Test WAVs

| Test | SNR | Duration | Noisy input | FPGA 8 kHz | FPGA 16 kHz |
| --- | ---: | ---: | --- | --- | --- |
| Bus | 2.5–17.5 dB | 23.682 s | [bus_noisy.wav](noisy/bus_noisy.wav) | [bus_fpga_8khz.wav](fpga8k/bus_fpga_8khz.wav) | [bus_fpga_16khz.wav](fpga16k/bus_fpga_16khz.wav) |
| Café | 2.5–17.5 dB | 20.509 s | [cafe_noisy.wav](noisy/cafe_noisy.wav) | [cafe_fpga_8khz.wav](fpga8k/cafe_fpga_8khz.wav) | [cafe_fpga_16khz.wav](fpga16k/cafe_fpga_16khz.wav) |
| Office | 2.5–17.5 dB | 21.980 s | [office_noisy.wav](noisy/office_noisy.wav) | [office_fpga_8khz.wav](fpga8k/office_fpga_8khz.wav) | [office_fpga_16khz.wav](fpga16k/office_fpga_16khz.wav) |
| Public square | 2.5–17.5 dB | 21.473 s | [psquare_noisy.wav](noisy/psquare_noisy.wav) | [psquare_fpga_8khz.wav](fpga8k/psquare_fpga_8khz.wav) | [psquare_fpga_16khz.wav](fpga16k/psquare_fpga_16khz.wav) |
| Bus | 2.5 dB | 20.879 s | [bus_low_snr_noisy.wav](noisy/bus_low_snr_noisy.wav) | [bus_low_snr_fpga_8khz.wav](fpga8k/bus_low_snr_fpga_8khz.wav) | [bus_low_snr_fpga_16khz.wav](fpga16k/bus_low_snr_fpga_16khz.wav) |
| Café | 2.5 dB | 21.466 s | [cafe_low_snr_noisy.wav](noisy/cafe_low_snr_noisy.wav) | [cafe_low_snr_fpga_8khz.wav](fpga8k/cafe_low_snr_fpga_8khz.wav) | [cafe_low_snr_fpga_16khz.wav](fpga16k/cafe_low_snr_fpga_16khz.wav) |
| Office | 2.5 dB | 21.494 s | [office_low_snr_noisy.wav](noisy/office_low_snr_noisy.wav) | [office_low_snr_fpga_8khz.wav](fpga8k/office_low_snr_fpga_8khz.wav) | [office_low_snr_fpga_16khz.wav](fpga16k/office_low_snr_fpga_16khz.wav) |
| Public square | 2.5 dB | 21.525 s | [psquare_low_snr_noisy.wav](noisy/psquare_low_snr_noisy.wav) | [psquare_low_snr_fpga_8khz.wav](fpga8k/psquare_low_snr_fpga_8khz.wav) | [psquare_low_snr_fpga_16khz.wav](fpga16k/psquare_low_snr_fpga_16khz.wav) |

Tests join complete VoiceBank-DEMAND utterances with 250-ms silence gaps.
The four 2.5-dB tests use additional source utterances. All WAVs are mono
16 kHz; the 8-kHz runner resamples on the host and restores the source rate
for playback. Sources and transformations: [attribution](ATTRIBUTION.md),
[input manifest](input_manifest.json).

## Performance

**173.010 seconds; 17,353 frames per model.**
Italy RK-256, build `0xdf0749de`, measured clock 333.25 MHz.
FPGA host: core 6; FP32 CPU reference: one thread on core 7.

| Measurement | Native 8 kHz | Existing 16 kHz |
| --- | ---: | ---: |
| Audio processing time | 171.253 s | 248.505 s |
| Audio RTF | **0.9898** | **1.4364** |
| CPU audio RTF | 0.0674 | 0.0704 |
| Mean FPGA / host time per frame | 9.070 / 9.849 ms | 13.447 / 14.310 ms |
| Host p99 / maximum | 10.255 / 15.047 ms | 14.722 / 21.585 ms |
| Host frames over 10 ms | 3,005 / 17,353 | 17,353 / 17,353 |
| FPGA frames over 10 ms | 0 / 17,353 | 17,353 / 17,353 |
| Waveform error vs CPU | 1.683% | 4.053% |

**RTF = processing time / audio duration; below 1 is faster than realtime.**
Audio timing includes framing, reconstruction and WAV I/O; it excludes
initial loading. All frames, including flush frames, are counted.
The 8-kHz model meets average realtime throughput with host deadline misses;
the 16-kHz model is slower than realtime.

All outputs are finite and preserve source sample counts. Waveform error
is pooled relative L2 against each model’s FP32 CPU output, without fitted
gain, alignment or trimming; it measures numerical agreement.
The existing 16-kHz accuracy issue remains open.

## FPGA resources

**Target FPGA: Kintex UltraScale+ KU5P.** Implementation figures supplied by the user:

| Andromeda Core resource | Count |
| --- | ---: |
| **Total LUTs** | **79,639** |
| Logic LUTs | 77,410 |
| LUTRAMs | 784 |
| SRLs | 1,445 |
| **Flip-flops** | **73,514** |
| RAMB36 | 19 |
| RAMB18 | 3 |
| URAM | 34 |
| DSP blocks | 136 |

Reported timing meets the **3.000 ns constraint** (approximately
**333.3 MHz**): **WNS +0.008 ns**, **TNS 0 ns**.
[Resource and timing record](fpga_resources.json).

## Execution and commands

Each model uses one bin: program, parameters and initial state upload to
DRAM once. Each 10-ms hop uses host STFT → input DRAM write → one
execute/HALT → output DRAM read → host iSTFT. Neural computation runs on
FPGA and recurrent state stays in DRAM. The runner compensates the
40-ms model delay.
[Bin sizes and execution audit](../../DUAL_RATE_BENCHMARK.md#verified-dram-execution-sequence).

From the repository root, with exclusive access to the FPGA:

```bash
source /home/hunlu/my_torch_env/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --device rk --dev xdma0 --cpu-core 6 \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav \
  --output /tmp/bus_low_snr_fpga_8khz.wav

python models/dpdfnet/dpdfnet_run_from_bin.py \
  --device rk --dev xdma0 --cpu-core 6 \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav \
  --output /tmp/bus_low_snr_fpga_16khz.wav
```

[Per-test results](results.json) · [Summary](summary.json) ·
[Independent audit](independent_audit.json) · [SHA-256 inventory](SHA256SUMS).
The [file map](published_artifacts.json) connects the new filenames to original
run paths. Logs and metrics retain their original case names, such as
`bus.log` and `bus.metrics.json`.
