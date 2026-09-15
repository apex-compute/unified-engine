# Matrix FIR accuracy comparison

FPGA reconstruction versus the frozen official FP32 CPU codec; no gain, delay or polarity fitting.

| Profile | Pooled waveform L2 | Matching tokens | Processing RTF | Startup-inclusive RTF |
| --- | ---: | ---: | ---: | ---: |
| serial_baseline | 23.941% | 12796/13845 | 4.82416 | 7.17662 |
| sorted_encoder | 22.640% | 13027/13845 | 4.82497 | 7.32377 |
| matrix_encoder | 23.847% | 13035/13845 | 6.40878 | 8.58756 |

Baseline: serial_baseline. CPU processing RTF: 2.32369. RTF 1 means real time. Pooling uses summed energies and duration-weighted measured time.

| Case | Profile | Waveform L2 | Tokens | Processing RTF | Startup-inclusive RTF | Waveform change |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bus | serial_baseline | 20.551% | 1736/1895 | 4.82395 | 7.14792 | unchanged |
| bus | sorted_encoder | 25.140% | 1758/1895 | 4.82500 | 7.40746 | regressed |
| bus | matrix_encoder | 20.360% | 1747/1895 | 6.40863 | 8.59258 | improved |
| cafe | serial_baseline | 21.331% | 1528/1641 | 4.82236 | 7.10369 | unchanged |
| cafe | sorted_encoder | 21.126% | 1557/1641 | 4.82333 | 7.19915 | improved |
| cafe | matrix_encoder | 19.001% | 1557/1641 | 6.40773 | 8.62912 | improved |
| office | serial_baseline | 22.478% | 1583/1759 | 4.82413 | 7.11737 | unchanged |
| office | sorted_encoder | 22.068% | 1623/1759 | 4.82581 | 7.30479 | improved |
| office | matrix_encoder | 23.031% | 1621/1759 | 6.40864 | 8.56466 | regressed |
| psquare | serial_baseline | 29.772% | 1569/1718 | 4.82391 | 7.19327 | unchanged |
| psquare | sorted_encoder | 21.482% | 1604/1718 | 4.82436 | 7.35811 | improved |
| psquare | matrix_encoder | 37.045% | 1614/1718 | 6.40768 | 8.57278 | regressed |
| bus_low_snr | serial_baseline | 21.442% | 1556/1671 | 4.82488 | 7.21242 | unchanged |
| bus_low_snr | sorted_encoder | 20.325% | 1583/1671 | 4.82602 | 7.37699 | improved |
| bus_low_snr | matrix_encoder | 16.941% | 1595/1671 | 6.40966 | 8.63227 | improved |
| cafe_low_snr | serial_baseline | 32.889% | 1622/1718 | 4.82503 | 7.24044 | unchanged |
| cafe_low_snr | sorted_encoder | 25.691% | 1641/1718 | 4.82534 | 7.34513 | improved |
| cafe_low_snr | matrix_encoder | 28.242% | 1637/1718 | 6.40970 | 8.57285 | improved |
| office_low_snr | serial_baseline | 17.179% | 1582/1720 | 4.82382 | 7.17816 | unchanged |
| office_low_snr | sorted_encoder | 15.908% | 1624/1720 | 4.82430 | 7.30949 | improved |
| office_low_snr | matrix_encoder | 15.888% | 1623/1720 | 6.40805 | 8.58195 | improved |
| psquare_low_snr | serial_baseline | 22.653% | 1620/1723 | 4.82517 | 7.22170 | unchanged |
| psquare_low_snr | sorted_encoder | 26.591% | 1637/1723 | 4.82552 | 7.27689 | regressed |
| psquare_low_snr | matrix_encoder | 22.346% | 1641/1723 | 6.41018 | 8.55743 | improved |

Regressions against the baseline:

- serial_baseline: waveform: none; tokens: none; processing: none.
- sorted_encoder: waveform: bus, psquare_low_snr; tokens: none; processing: bus, cafe, office, psquare, bus_low_snr, cafe_low_snr, office_low_snr, psquare_low_snr.
- matrix_encoder: waveform: office, psquare; tokens: none; processing: bus, cafe, office, psquare, bus_low_snr, cafe_low_snr, office_low_snr, psquare_low_snr.

Each FPGA run has one resident upload, one input upload, one START, one HALT, one output read, and zero CPU neural operations.

Startup-inclusive RTF includes runner artifact validation/loading and model upload. It excludes compilation, Python startup and final metrics serialization. Bin hashes are runner-recorded unless the JSON explicitly says the file was rehashed. No profile is selected automatically.

Matrix versus sorted encoder:

- waveform regressions: office, psquare, cafe_low_snr.
- tokens regressions: bus, office, cafe_low_snr, office_low_snr.
- processing regressions: bus, cafe, office, psquare, bus_low_snr, cafe_low_snr, office_low_snr, psquare_low_snr.

The matrix profile uses original FIR coefficients split into BF16 high and residual values, combined through native BF19/BF20 dot accumulation before a BF16 store. It changes both encoder upsampling and downsampling filters; decoder FIR and Snake remain legacy. Convolution, activation storage, LSTM input and recurrent weights are BF16. The decoder retains compensated LSTM math and the centered split codebook.

Serial baseline runs used build `0x40519e0a`; sorted and matrix runs used `0x90f1f464`. All use AXI 256 and the measured 333.25 MHz clock. These are whole-utterance codec reconstructions, not independent 10 ms frames or denoising quality scores.

The matrix run files below are exact copies of the original runner outputs. All eight matrix deployment bins, resident images, parameter prefixes and instruction sections were rehashed from the local deployment cache during packaging and matched the [compile catalog](matrix_compile_catalog.json). The standard report identifies bin hashes as runner-recorded; the additional `matrix_audit` and per-case `packaging_bin_hash_verification` fields record this separate rehash. Older profile bins were not rehashed during matrix packaging. Parameters include weights, biases, FIR matrices, constants, lookup tables and allocation padding.

| Case | Parameter bytes | Instruction bytes | Instructions | Resident bytes | Bin bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| bus | 333,627,264 | 637,939,776 | 19,935,618 | 971,567,040 | 971,608,623 |
| cafe | 333,627,264 | 553,391,040 | 17,293,470 | 887,018,304 | 887,059,887 |
| office | 333,627,264 | 592,654,784 | 18,520,462 | 926,282,048 | 926,323,631 |
| psquare | 333,627,264 | 579,132,928 | 18,097,904 | 912,760,192 | 912,801,775 |
| bus_low_snr | 333,627,264 | 562,815,296 | 17,587,978 | 896,442,560 | 896,484,143 |
| cafe_low_snr | 333,627,264 | 579,132,928 | 18,097,904 | 912,760,192 | 912,801,775 |
| office_low_snr | 333,627,264 | 579,201,536 | 18,100,048 | 912,828,800 | 912,870,383 |
| psquare_low_snr | 333,627,264 | 580,674,752 | 18,146,086 | 914,302,016 | 914,343,599 |

Exact SHA-256 section hashes:

| Case | Parameters SHA-256 | Instructions SHA-256 | Bin SHA-256 |
| --- | --- | --- | --- |
| bus | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `87edaa8ef6f80d6cbb16d3c9384443a4348ec67b531a32b59522eae3a3cb2464` | `61fc828ba2f8019f1227ec5c8af278fdab1498e194af5d7500419dbd1a767d32` |
| cafe | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `36de5d1f797ae0012c739fbcd0f4229f3a860649bcc532cc51f397593f0aad8e` | `e9e3155b75154844d845dfacccea70c35a275481070febc7ff05a6bce887dcb2` |
| office | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `cbeb1056581349435fead4876ac4580c0a4724ef12a80c8155b574227d122234` | `c6145af17973f73cf79d87a22ebe43b8bb383553ab7e8b8eeb21ad8acd181b25` |
| psquare | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `933594335478b7fc3e1f6d7e9cd585055b4c62b6f9bbfdf01bd0eecabe99030d` | `41f334e2fcfa9360c03cee24547f6ccb88d2a995fbf2b8731600d207a2fd74c2` |
| bus_low_snr | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `5490f47c5cba2f9366e130dec1f7e9c1b81c3363c706926d05d21dbdcfca3ea4` | `c71a097952ed39bb09c08e2165b46e123740d76512bea7a143fc13aa38634b22` |
| cafe_low_snr | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `933594335478b7fc3e1f6d7e9cd585055b4c62b6f9bbfdf01bd0eecabe99030d` | `dbda51ddfdb97d777455621fd14fc1cc2a8fe1193bbcf0bd8c4652fe68ded010` |
| office_low_snr | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `30f67e1a6e6f1cbe9d1b57684d214f1861d6bd1e173c2f924b9b109fe2729847` | `34bb9d68c2cb50d8e396168c7013c0f4c75ecbd0416b83e35d5c95271be9bb98` |
| psquare_low_snr | `e7adbc9ad660945b61d7b6910086831b240d2a5df6cd697484c3c7fc016f4530` | `5505a68e105439788550feeafbd53c061fafc465190d7db85b293f2dbd39346f` | `4ca16802350f17a48b016f864773dd89d3daf1409fd5612a71b504b05474a275` |

| Case | Input | CPU | Prior FPGA | Sorted FPGA | Matrix FPGA |
| --- | --- | --- | --- | --- | --- |
| bus | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav) | [WAV](../20260914_noisy20s/cpu/bus.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/bus.wav) | [WAV](sorted/bus.wav) | [WAV](matrix/bus.wav) |
| cafe | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_noisy.wav) | [WAV](../20260914_noisy20s/cpu/cafe.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/cafe.wav) | [WAV](sorted/cafe.wav) | [WAV](matrix/cafe.wav) |
| office | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/office_noisy.wav) | [WAV](../20260914_noisy20s/cpu/office.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/office.wav) | [WAV](sorted/office.wav) | [WAV](matrix/office.wav) |
| psquare | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_noisy.wav) | [WAV](../20260914_noisy20s/cpu/psquare.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/psquare.wav) | [WAV](sorted/psquare.wav) | [WAV](matrix/psquare.wav) |
| bus_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/bus_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/bus_low_snr.wav) | [WAV](sorted/bus_low_snr.wav) | [WAV](matrix/bus_low_snr.wav) |
| cafe_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/cafe_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/cafe_low_snr.wav) | [WAV](sorted/cafe_low_snr.wav) | [WAV](matrix/cafe_low_snr.wav) |
| office_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/office_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/office_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/office_low_snr.wav) | [WAV](sorted/office_low_snr.wav) | [WAV](matrix/office_low_snr.wav) |
| psquare_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/psquare_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/psquare_low_snr.wav) | [WAV](sorted/psquare_low_snr.wav) | [WAV](matrix/psquare_low_snr.wav) |

[Exact metrics, section hashes and regressions](matrix_results.json), [frozen inputs](matrix_manifest.json). The standard offline report command recomputes the paired WAV/token and runtime results using tracked artifacts alone. Cached deployment bins are not required. The additional `matrix_sections`, `matrix_audit` and `matrix_against_sorted` fields were produced during packaging; the standard report does not regenerate those extensions or rehash binary sections.

```bash
python models/bigcodec/bigcodec_accuracy_report.py \
  models/bigcodec/validation/20260915_encoder_accuracy/matrix_manifest.json
```
