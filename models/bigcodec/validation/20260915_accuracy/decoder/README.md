# BigCodec noisy-audio accuracy comparison

BigCodec reconstructs audio; it is not trained for background-noise suppression. These errors measure FPGA agreement with the CPU codec, not noise removal. See the [background-noise implementation check](../filtering_check/README.md).

16/16 runs completed across eight noisy test cases. Results compare FPGA reconstructions with the frozen official FP32 CPU reconstructions.

| Recurrent weights | Processing RTF, before → after | Pooled waveform error, before → after | Token agreement, before → after | Cases improved / regressed / unchanged |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 4.7965 → 4.8242 | 28.057% → 23.941% | 12607/13845 → 12796/13845 | 8 / 0 / 0 |
| IF8 | 4.3555 → 4.3827 | 27.040% → 24.350% | 12617/13845 → 12778/13845 | 7 / 1 / 0 |

Frozen single-thread FP32 CPU processing RTF: 2.32369 (402.021 s / 173.00981 s audio). RTF 1 means real time; lower is faster. Each before/after aggregate uses the same cases. Waveform pooling sums error and reference energies.

BF16 convolutions and activations are used throughout. Compensated cell state/tanh and fused gates apply to the decoder LSTM; the encoder retains its original arithmetic. Codebook scores are centered and codebook weights retain BF16 residuals.

| Case | Weights | Waveform error, before → after | Tokens equal, before → after | Processing seconds, before → after | RTF, before → after |
| --- | --- | ---: | ---: | ---: | ---: |
| bus | BF16 | 28.090% → 20.551% | 1714 → 1736 / 1895 | 113.591 → 114.242 | 4.7964 → 4.8240 |
| bus | IF8 | 27.451% → 22.220% | 1708 → 1738 / 1895 | 103.136 → 103.780 | 4.3550 → 4.3822 |
| cafe | BF16 | 21.961% → 21.331% | 1503 → 1528 / 1641 | 98.335 → 98.902 | 4.7947 → 4.8224 |
| cafe | IF8 | 24.750% → 22.153% | 1514 → 1520 / 1641 | 89.296 → 89.851 | 4.3540 → 4.3810 |
| office | BF16 | 23.844% → 22.478% | 1576 → 1583 / 1759 | 105.428 → 106.037 | 4.7965 → 4.8241 |
| office | IF8 | 22.177% → 22.886% | 1567 → 1580 / 1759 | 95.738 → 96.333 | 4.3556 → 4.3827 |
| psquare | BF16 | 33.449% → 29.772% | 1546 → 1569 / 1718 | 102.979 → 103.586 | 4.7957 → 4.8239 |
| psquare | IF8 | 34.861% → 29.280% | 1538 → 1566 / 1718 | 93.515 → 94.100 | 4.3549 → 4.3822 |
| bus_low_snr | BF16 | 22.796% → 21.442% | 1529 → 1556 / 1671 | 100.165 → 100.739 | 4.7974 → 4.8249 |
| bus_low_snr | IF8 | 24.141% → 21.513% | 1531 → 1553 / 1671 | 90.953 → 91.524 | 4.3562 → 4.3835 |
| cafe_low_snr | BF16 | 36.989% → 32.889% | 1598 → 1622 / 1718 | 102.983 → 103.572 | 4.7976 → 4.8250 |
| cafe_low_snr | IF8 | 35.994% → 33.499% | 1596 → 1620 / 1718 | 93.513 → 94.095 | 4.3564 → 4.3835 |
| office_low_snr | BF16 | 19.636% → 17.179% | 1551 → 1582 / 1720 | 103.098 → 103.685 | 4.7965 → 4.8238 |
| office_low_snr | IF8 | 19.879% → 17.013% | 1571 → 1582 / 1720 | 93.620 → 94.206 | 4.3556 → 4.3828 |
| psquare_low_snr | BF16 | 32.177% → 22.653% | 1590 → 1620 / 1723 | 103.268 → 103.864 | 4.7975 → 4.8252 |
| psquare_low_snr | IF8 | 24.162% → 23.079% | 1592 → 1619 / 1723 | 93.773 → 94.356 | 4.3564 → 4.3835 |

Each complete WAV uses one program/parameter upload, one input upload, one START, one HALT and one output bundle read. Completed runs have finite output, zero padding and zero CPU neural operations. The audio-processing RTF excludes compilation, artifact loading and the resident upload; JSON/CSV also include total runner latency.

Including runner startup, pooled RTF changes from **6.57 to 7.18 for BF16** and **6.04 to 6.67 for IF8**. The larger instruction programs increase startup cost. Real time requires RTF ≤ 1.

Platform: Italy, Kintex UltraScale+ KU5P; build 0x40519e0a, AXI 256, 333.25 MHz, 2 GiB visible DRAM. These files contain noisy speech, but BigCodec is an audio codec. Waveform error measures agreement with CPU reconstruction, not noise removal or perceptual quality.

[Per-case metrics and deployment sizes](metrics_table.csv) · [Machine-readable summary and evidence hashes](metrics_summary.json) · [Frozen CPU references and source attribution](../../20260914_noisy20s/README.md) · [Bus pilot error decomposition](../bus_error_decomposition.md)

## Deployment sizes

| Recurrent weights | Parameter bytes | Program bytes | Instructions | Resident bytes |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 331,267,968 | 589,888,192–681,269,056 | 18,434,006–21,289,658 | 921,156,160–1,012,537,024 |
| IF8 | 294,698,880 | 572,244,160–660,894,016 | 17,882,630–20,652,938 | 866,943,040–955,592,896 |

Each bin contains both the parameters and instruction program. Sizes vary with the padded audio length; hashes and exact per-file sizes are in the CSV/JSON.

## Audio

| Case | Duration | Noisy input | Official CPU | FPGA BF16 | FPGA IF8 |
| --- | ---: | --- | --- | --- | --- |
| bus | 23.682 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/bus.wav) | [WAV](fpga_bf16/bus.wav) | [WAV](fpga_if8/bus.wav) |
| cafe | 20.509 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/cafe.wav) | [WAV](fpga_bf16/cafe.wav) | [WAV](fpga_if8/cafe.wav) |
| office | 21.980 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/office_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/office.wav) | [WAV](fpga_bf16/office.wav) | [WAV](fpga_if8/office.wav) |
| psquare | 21.473 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/psquare.wav) | [WAV](fpga_bf16/psquare.wav) | [WAV](fpga_if8/psquare.wav) |
| bus_low_snr | 20.879 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/bus_low_snr.wav) | [WAV](fpga_bf16/bus_low_snr.wav) | [WAV](fpga_if8/bus_low_snr.wav) |
| cafe_low_snr | 21.466 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/cafe_low_snr.wav) | [WAV](fpga_bf16/cafe_low_snr.wav) | [WAV](fpga_if8/cafe_low_snr.wav) |
| office_low_snr | 21.494 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/office_low_snr_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/office_low_snr.wav) | [WAV](fpga_bf16/office_low_snr.wav) | [WAV](fpga_if8/office_low_snr.wav) |
| psquare_low_snr | 21.525 s | [WAV](../../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_low_snr_noisy.wav) | [WAV](../../20260914_noisy20s/cpu/psquare_low_snr.wav) | [WAV](fpga_bf16/psquare_low_snr.wav) | [WAV](fpga_if8/psquare_low_snr.wav) |

[Token-selection and conditional decoder errors](conditional_error_decomposition.md) · [Short-clip regression](../short_clip/README.md) · [Reproduce the measurements](reproduction.md)
