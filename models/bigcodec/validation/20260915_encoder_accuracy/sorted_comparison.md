# Sorted encoder FIR comparison

Sorted FIR is an optional candidate and is not the default. The change affects encoder FIR tap accumulation order only. The eight-file waveform error improves in aggregate, with regressions on **bus** and **psquare_low_snr**.

FPGA reconstruction versus the frozen official FP32 CPU codec; no gain, delay or polarity fitting.

| Profile | Pooled waveform L2 | Matching tokens | Processing RTF | Startup-inclusive RTF |
| --- | ---: | ---: | ---: | ---: |
| serial_baseline | 23.941% | 12796/13845 | 4.82416 | 7.17662 |
| sorted_encoder | 22.640% | 13027/13845 | 4.82497 | 7.32377 |

Baseline: serial_baseline. CPU processing RTF: 2.32369. RTF 1 means real time. Pooling uses summed energies and duration-weighted measured time.

| Case | Profile | Waveform L2 | Tokens | Processing RTF | Startup-inclusive RTF | Waveform change |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| bus | serial_baseline | 20.551% | 1736/1895 | 4.82395 | 7.14792 | unchanged |
| bus | sorted_encoder | 25.140% | 1758/1895 | 4.82500 | 7.40746 | regressed |
| cafe | serial_baseline | 21.331% | 1528/1641 | 4.82236 | 7.10369 | unchanged |
| cafe | sorted_encoder | 21.126% | 1557/1641 | 4.82333 | 7.19915 | improved |
| office | serial_baseline | 22.478% | 1583/1759 | 4.82413 | 7.11737 | unchanged |
| office | sorted_encoder | 22.068% | 1623/1759 | 4.82581 | 7.30479 | improved |
| psquare | serial_baseline | 29.772% | 1569/1718 | 4.82391 | 7.19327 | unchanged |
| psquare | sorted_encoder | 21.482% | 1604/1718 | 4.82436 | 7.35811 | improved |
| bus_low_snr | serial_baseline | 21.442% | 1556/1671 | 4.82488 | 7.21242 | unchanged |
| bus_low_snr | sorted_encoder | 20.325% | 1583/1671 | 4.82602 | 7.37699 | improved |
| cafe_low_snr | serial_baseline | 32.889% | 1622/1718 | 4.82503 | 7.24044 | unchanged |
| cafe_low_snr | sorted_encoder | 25.691% | 1641/1718 | 4.82534 | 7.34513 | improved |
| office_low_snr | serial_baseline | 17.179% | 1582/1720 | 4.82382 | 7.17816 | unchanged |
| office_low_snr | sorted_encoder | 15.908% | 1624/1720 | 4.82430 | 7.30949 | improved |
| psquare_low_snr | serial_baseline | 22.653% | 1620/1723 | 4.82517 | 7.22170 | unchanged |
| psquare_low_snr | sorted_encoder | 26.591% | 1637/1723 | 4.82552 | 7.27689 | regressed |

Regressions against the baseline:

- serial_baseline: waveform: none; tokens: none; processing: none.
- sorted_encoder: waveform: bus, psquare_low_snr; tokens: none; processing: bus, cafe, office, psquare, bus_low_snr, cafe_low_snr, office_low_snr, psquare_low_snr.

Each FPGA run has one resident upload, one input upload, one START, one HALT, one output read, and zero CPU neural operations.

Startup-inclusive RTF includes runner artifact validation/loading and model upload. It excludes compilation, Python startup and final metrics serialization. Bin hashes are runner-recorded unless the JSON explicitly says the file was rehashed. No profile is selected automatically.

The separate 3.956 s speech clip is excluded from the eight-file pool: waveform error worsens from **20.0069% to 31.7715%**, with 310 to 303/317 matching tokens. [Prior FPGA WAV](../20260915_accuracy/short_clip/audio/split_decoder_fpga.wav), [sorted FPGA WAV](short/sorted.wav), [CPU WAV](../20260915_accuracy/short_clip/audio/official_cpu.wav), [short comparison](short/sorted.comparison.json).

Before uses build `0x40519e0a`; sorted uses `0x90f1f464`. The original run records and comparison paths are preserved verbatim; the frozen manifest binds the copied artifacts at their current locations. Both sets use AXI 256 and a measured 333.25 MHz clock. Six sorted records additionally spell out the unchanged legacy Snake defaults; the other two use the earlier metadata schema.

| Case | Input | CPU | Prior FPGA | Sorted FPGA |
| --- | --- | --- | --- | --- |
| bus | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav) | [WAV](../20260914_noisy20s/cpu/bus.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/bus.wav) | [WAV](sorted/bus.wav) |
| cafe | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_noisy.wav) | [WAV](../20260914_noisy20s/cpu/cafe.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/cafe.wav) | [WAV](sorted/cafe.wav) |
| office | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/office_noisy.wav) | [WAV](../20260914_noisy20s/cpu/office.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/office.wav) | [WAV](sorted/office.wav) |
| psquare | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_noisy.wav) | [WAV](../20260914_noisy20s/cpu/psquare.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/psquare.wav) | [WAV](sorted/psquare.wav) |
| bus_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/bus_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/bus_low_snr.wav) | [WAV](sorted/bus_low_snr.wav) |
| cafe_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/cafe_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/cafe_low_snr.wav) | [WAV](sorted/cafe_low_snr.wav) |
| office_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/office_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/office_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/office_low_snr.wav) | [WAV](sorted/office_low_snr.wav) |
| psquare_low_snr | [WAV](../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_low_snr_noisy.wav) | [WAV](../20260914_noisy20s/cpu/psquare_low_snr.wav) | [WAV](../20260915_accuracy/decoder/fpga_bf16/psquare_low_snr.wav) | [WAV](sorted/psquare_low_snr.wav) |

[Exact metrics and hashes](sorted_results.json), [frozen input manifest](sorted_manifest.json), [historical run-source hashes](sorted_batch_sources.json), [original batch results](sorted_batch_results.json).

Recompute the JSON without changing recorded WAVs or runtime metrics:

```bash
python models/bigcodec/bigcodec_accuracy_report.py \
  models/bigcodec/validation/20260915_encoder_accuracy/sorted_manifest.json
```
