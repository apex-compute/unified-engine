# Paired decoder validation

Eight noisy files: pooled full-codec waveform error **23.94% → 23.91%**; decoder error with identical tokens **3.71% → 3.09%**. Processing RTF **4.8242 → 4.8790**. RTF is processing time/audio duration; below 1 is realtime.

The full-codec comparison uses the official CPU encoder and decoder. The conditional comparison decodes the unchanged FPGA tokens on the CPU, isolating decoder arithmetic. Conditional error is not a replacement for the full-codec target.

Case links open the new FPGA output. Audio references link the noisy input and official CPU reconstruction.

| Case (FPGA output) | Audio references | Full-codec L2 old → new | Same-token decoder L2 old → new | RTF old → new | CPU token matches |
| --- | --- | ---: | ---: | ---: | ---: |
| [bus](fpga/bus.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/bus.wav) | 20.551% → 19.903% | 4.628% → 2.959% | 4.8240 → 4.8786 | 1736/1895 |
| [cafe](fpga/cafe.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/cafe.wav) | 21.331% → 21.266% | 2.792% → 2.847% | 4.8224 → 4.8775 | 1528/1641 |
| [office](fpga/office.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/office_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/office.wav) | 22.478% → 22.384% | 3.503% → 3.069% | 4.8241 → 4.8791 | 1583/1759 |
| [psquare](fpga/psquare.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/psquare.wav) | 29.772% → 30.164% | 3.688% → 3.078% | 4.8239 → 4.8783 | 1569/1718 |
| [bus_low_snr](fpga/bus_low_snr.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/bus_low_snr.wav) | 21.442% → 20.887% | 3.979% → 3.205% | 4.8249 → 4.8796 | 1556/1671 |
| [cafe_low_snr](fpga/cafe_low_snr.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/cafe_low_snr_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/cafe_low_snr.wav) | 32.889% → 33.098% | 3.667% → 3.081% | 4.8250 → 4.8800 | 1622/1718 |
| [office_low_snr](fpga/office_low_snr.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/office_low_snr_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/office_low_snr.wav) | 17.179% → 17.041% | 3.466% → 3.170% | 4.8238 → 4.8791 | 1582/1720 |
| [psquare_low_snr](fpga/psquare_low_snr.wav) | [Input](../../../../dpdfnet/validation/20260914_noisy20s/noisy/psquare_low_snr_noisy.wav) · [CPU](../../20260914_noisy20s/cpu/psquare_low_snr.wav) | 22.653% → 22.775% | 3.805% → 3.306% | 4.8252 → 4.8803 | 1620/1723 |

All eight full errors remain above 10%; all nine conditional errors are below 10% (including the separate short clip).
Raw-error regressions: psquare, cafe_low_snr, psquare_low_snr. Conditional-error regressions: cafe.

On psquare, lower decoder error removes some accidental cancellation between encoder-token and decoder errors: the negative cross term becomes smaller in magnitude, so full-codec error rises. The exact energy identity is recomputed in the JSON.

| Eight-file arithmetic mean | Old FPGA | New FPGA |
| --- | ---: | ---: |
| PESQ-WB vs CPU | 4.021986 | 4.025388 |
| STOI vs CPU | 0.979003 | 0.979017 |
| PESQ-WB codec-quality gap | -0.012667 | -0.010964 |
| STOI codec-quality gap | -0.001958 | -0.001882 |

Codec-quality gap is FPGA minus CPU score against the same original noisy input. CPU baseline: PESQ-WB 1.814827, STOI 0.819734. These noisy-reference scores describe codec fidelity, not denoising or word accuracy. PESQ retains its standard internal alignment/level normalization; no external delay, gain, or polarity fitting is applied. Raw L2 uses original sample indices and rate. Speech/spectral metrics use shared 16 kHz resampling. [Metric definitions and primary sources](../perceptual/sources.md).

Independent [short-clip FPGA output](fpga/p232_007.wav) (3.96 s, 48 kHz, excluded from the eight-file pool): full L2 20.007% → 16.228%; same-token decoder L2 12.287% → 6.122%; RTF 4.9053 → 4.9232. 310/317 tokens match the CPU; all 317 are unchanged from the old FPGA.

| Compiled input | Old program bytes | New program bytes | New parameter bytes | New resident bytes |
| --- | ---: | ---: | ---: | ---: |
| p232_007 | 114,322,432 | 70,428,352 | 331,267,968 | 401,696,320 |
| bus | 681,269,056 | 415,697,536 | 331,267,968 | 746,965,504 |
| cafe | 589,888,192 | 359,998,592 | 331,267,968 | 691,266,560 |
| office | 632,431,808 | 385,965,568 | 331,267,968 | 717,233,536 |
| psquare | 617,762,688 | 377,056,128 | 331,267,968 | 708,324,096 |
| bus_low_snr | 600,779,136 | 366,675,136 | 331,267,968 | 697,943,104 |
| cafe_low_snr | 617,762,688 | 377,056,128 | 331,267,968 | 708,324,096 |
| office_low_snr | 618,324,608 | 377,337,088 | 331,267,968 | 708,605,056 |
| psquare_low_snr | 619,552,640 | 378,143,680 | 331,267,968 | 709,411,648 |

Eight-file native RTF: 4.8131 → 4.8683; startup-inclusive RTF: 7.1766 → 6.3353. RTFs are total measured time divided by total audio duration. Compiled programs cover whole files, not 10 ms streaming chunks.

All nine new runs use build `0x90f1f464`, AXI256, BF16 convolution/recurrent weights, compensated decoder cell/tanh and paired sigmoid. The encoder retains native BF16 sigmoid. Each saved receipt has one model upload, one input upload, one START, one HALT and one output read, zero CPU neural operations, finite output and zero-valued padding. Encoder token IDs and metadata are unchanged. Long-file historical baseline receipts use build `0x40519e0a`; the short baseline uses the current build. A [current-image bus control](../loop/bus_baseline_image_parity.json) records identical decoded sample bits and tokens for the old binary on both builds; the reproducer audits that receipt.

The reproducer rehashes tracked WAVs, tokens, CPU provenance, native records and frozen source files; recomputes raw/conditional errors, new perceptual scores and timing aggregates; and verifies the frozen historical CPU-quality baseline. Bin SHA256 and exact program/parameter sizes are runner-recorded in [results.json](results.json). Reproduction does not require cached binaries, model weights, or hardware and does not rerun inference.

```bash
pip install -r models/bigcodec/requirements-quality.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python models/bigcodec/validation/20260915_comparison_criteria/final/reproduce.py \
  --output /tmp/bigcodec-final-results.json --markdown /tmp/bigcodec-final-report.md
```
