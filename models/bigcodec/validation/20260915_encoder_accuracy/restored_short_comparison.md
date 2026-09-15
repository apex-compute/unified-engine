# BigCodec accuracy comparison

FPGA reconstruction versus the frozen official FP32 CPU codec; no gain, delay or polarity fitting.

| Profile | Pooled waveform L2 | Matching tokens | Processing RTF | Startup-inclusive RTF |
| --- | ---: | ---: | ---: | ---: |
| baseline_restored | 20.007% | 310/317 | 4.90527 | 7.46259 |
| matrix_production | 16.104% | 307/317 | 6.45730 | 8.89772 |
| matrix_bf16_rejected | 33.775% | 301/317 | 5.83069 | 8.26549 |

Baseline: baseline_restored. CPU processing RTF: 1.63947. RTF 1 means real time. Pooling uses summed energies and duration-weighted measured time.

| Case | Profile | Waveform L2 | Tokens | Processing RTF | Startup-inclusive RTF | Waveform change |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p232_007 | baseline_restored | 20.007% | 310/317 | 4.90527 | 7.46259 | unchanged |
| p232_007 | matrix_production | 16.104% | 307/317 | 6.45730 | 8.89772 | improved |
| p232_007 | matrix_bf16_rejected | 33.775% | 301/317 | 5.83069 | 8.26549 | regressed |

Regressions against the baseline:

- baseline_restored: waveform: none; tokens: none; processing: none.
- matrix_production: waveform: none; tokens: p232_007; processing: p232_007.
- matrix_bf16_rejected: waveform: p232_007; tokens: p232_007; processing: p232_007.

Each FPGA run has one resident upload, one input upload, one START, one HALT, one output read, and zero CPU neural operations.

Startup-inclusive RTF includes runner artifact validation/loading and model upload. It excludes compilation, Python startup and final metrics serialization. Bin hashes are runner-recorded unless the JSON explicitly says the file was rehashed. No profile is selected automatically.
