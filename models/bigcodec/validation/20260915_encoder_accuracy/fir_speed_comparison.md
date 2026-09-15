# FIR speed experiments

These are optional, unpromoted experiments on one 3.956 s clip. The compact variant is worse and slower than production matrix FIR. The fused writeback variant preserves every decoded sample bit and token, with a 1.508% reduction in measured processing time. It remains above RTF 6; neither experiment changes the production implementation or default profile.

FPGA reconstruction versus the frozen official FP32 CPU codec; no gain, delay or polarity fitting.

| Profile | Pooled waveform L2 | Matching tokens | Processing RTF | Startup-inclusive RTF |
| --- | ---: | ---: | ---: | ---: |
| matrix_production | 16.104% | 307/317 | 6.45730 | 8.89772 |
| matrix_compact | 19.142% | 308/317 | 6.58413 | 9.02510 |
| matrix_fused_write | 16.104% | 307/317 | 6.35992 | 8.78495 |

Baseline: matrix_production. CPU processing RTF: 1.63947. RTF 1 means real time. Pooling uses summed energies and duration-weighted measured time.

| Case | Profile | Waveform L2 | Tokens | Processing RTF | Startup-inclusive RTF | Waveform change |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p232_007 | matrix_production | 16.104% | 307/317 | 6.45730 | 8.89772 | unchanged |
| p232_007 | matrix_compact | 19.142% | 308/317 | 6.58413 | 9.02510 | regressed |
| p232_007 | matrix_fused_write | 16.104% | 307/317 | 6.35992 | 8.78495 | unchanged |

Regressions against the baseline:

- matrix_production: waveform: none; tokens: none; processing: none.
- matrix_compact: waveform: p232_007; tokens: none; processing: p232_007.
- matrix_fused_write: waveform: none; tokens: none; processing: none.

Each FPGA run has one resident upload, one input upload, one START, one HALT, one output read, and zero CPU neural operations.

Startup-inclusive RTF includes runner artifact validation/loading and model upload. It excludes compilation, Python startup and final metrics serialization. Bin hashes are runner-recorded unless the JSON explicitly says the file was rehashed. No profile is selected automatically.

The compact variant uses 32-output FIR tiles (up K64, down K192), changing native reduction grouping. The fused writeback variant retains 64-output tiles (up K128, down K384) and combines duplicate transpose writeback. Both retain the original split BF16 FIR coefficients. These findings are limited to this clip; the eight-file matrix results remain separate.

| Profile | Parameter bytes | Instruction bytes | Bin SHA-256 | WAV |
| --- | ---: | ---: | --- | --- |
| matrix_production | 333,627,264 | 110,352,448 | `53d7e7eb89e5ae821248502de47497c447b9c5657c5030d32fd26131fef2f5d8` | [WAV](short/matrix_production.wav) |
| matrix_compact | 331,857,792 | 109,985,408 | `c55ac7f9a6f238dd04020fc2dcf8f37edf7fd3eebead7625d906509ace2461ee` | [WAV](short/matrix_compact.wav) |
| matrix_fused_write | 333,627,264 | 110,335,744 | `24cdb44dd6d0aea9b6915ebc2004a43126e0a4311a34118a23267a689a4c7f02` | [WAV](short/matrix_fused_write.wav) |

[Decision and copy audit](diagnostics/fir_speed_decision.json), [verified source hashes](diagnostics/fir_speed_source_hashes.json), [compact compile](diagnostics/fir_speed_matrix_compact.compile.json), [compact pilot](diagnostics/fir_speed_matrix_compact.pilot.json), [fused compile](diagnostics/fir_speed_matrix_fused_write.compile.json), [fused pilot](diagnostics/fir_speed_matrix_fused_write.pilot.json). The copied receipts preserve their original paths. Experimental cached bins and compile sources were rehashed during packaging; the standard report records runner bin hashes and recomputes the output/timing comparisons from tracked artifacts without cached bins.

```bash
python models/bigcodec/bigcodec_accuracy_report.py \
  models/bigcodec/validation/20260915_encoder_accuracy/fir_speed_manifest.json
```
