# BigCodec numerical accuracy

The corrected profile uses compensated decoder LSTM cell/tanh arithmetic, fused gate projections, centered codebook scores and BF16 codebook weight residuals. The encoder LSTM retains the original arithmetic. Model structure and checkpoint are unchanged; all neural operations run on the existing FPGA bitstream.

[Eight-file BF16/IF8 comparison and reconstructed audio](decoder/README.md) · [Numerical diagnosis](debugging.md) · [Software tests](software_tests.log)

All sixteen runs completed on the original eight 20–24-second noisy files:

| Recurrent weights | Pooled waveform error, before → after | Processing RTF, before → after | Files improved |
| --- | ---: | ---: | ---: |
| BF16 | 28.057% → 23.941% | 4.7965 → 4.8242 | 8 / 8 |
| IF8 | 27.040% → 24.350% | 4.3555 → 4.3827 | 7 / 8 |

Relative waveform error falls by **14.7% for BF16** and **9.9% for IF8**. IF8 office error rises from 22.177% to 22.886%. Including runner startup, RTF rises from 6.57 to 7.18 for BF16 and 6.04 to 6.67 for IF8. Parameter and instruction sizes are in the full comparison.

The 23.682-second bus pilot selected the decoder-only setting:

| BF16 setting | Waveform error vs official CPU | Processing RTF | Matching tokens |
| --- | ---: | ---: | ---: |
| Original | 28.090% | 4.796 | 1,714 / 1,895 |
| Corrections in both LSTM stacks | 24.708% | 4.851 | 1,725 / 1,895 |
| Corrections in decoder LSTM only | 20.551% | 4.824 | 1,736 / 1,895 |

The [bus decomposition](bus_error_decomposition.md) separates token-selection and conditional decoder errors. The [fixed-input native LSTM test](lstm_probe/README.md) measures local error falling from 2.1353% to 0.9541%, independently of encoder/token changes.

The [3.956-second diagnostic clip](short_clip/README.md) regresses from 19.29% to 20.01% with the selected profile. Its original token-selection and decoder errors strongly canceled; smaller component errors do not guarantee lower total waveform error on every recording.

Waveform error is relative L2 at the original sample indices, without delay, gain or polarity fitting. It measures agreement with the official FP32 codec, not denoising or perceptual quality. RTF is processing seconds divided by audio seconds; real time requires RTF ≤ 1. Loading and upload are reported separately.

Each full audio file uses one resident program/parameter upload, one input upload, one START, one terminal HALT and one output-bundle read from DRAM. The model processes a complete utterance, with a 12.5 ms token interval; these are not independent 10 ms calls. Platform: Italy, Kintex UltraScale+ KU5P, RK AXI256, build `0x40519e0a`, 333.25 MHz, 2 GiB visible DRAM.

Use the [documented compile and run command](../../README.md#audio-file-to-audio-file-on-fpga). The numerical options are explicit; omitting them retains the original arithmetic. Bins contain parameters and instructions and remain in the ignored deployment directory. Per-file sizes, hashes, timings and execution counts are in the full comparison.

The corrected BF16 default bin is prepared on Italy for the short diagnostic input; [deployment.json](deployment.json) identifies it and the preserved original. The [fresh-run reproduction guide](decoder/reproduction.md) covers all sixteen noisy-file measurements.

The root `batch_freeze.json`, `hardware_runs.json` and `fpga_bf16/bus.*` retain the both-stack pilot. Its IF8 image was compiled but not executed after the decoder-only setting won the pilot. The selected sixteen-run batch is recorded separately under `decoder/`.
