# BigCodec accuracy comparison

FPGA reconstruction versus the frozen official FP32 CPU codec; no gain, delay or polarity fitting.

| Profile | Pooled waveform L2 | Matching tokens | Processing RTF | Startup-inclusive RTF |
| --- | ---: | ---: | ---: | ---: |
| baseline_current_board | 20.007% | 310/317 | 4.86711 | 7.50948 |
| sorted | 31.772% | 303/317 | 4.87308 | 7.60455 |
| matrix | 16.104% | 307/317 | 6.45810 | 8.97241 |
| matrix_up | 26.039% | 307/317 | 5.65559 | 8.30981 |
| matrix_down | 28.495% | 307/317 | 5.67323 | 8.27455 |

Baseline: baseline_current_board. CPU processing RTF: 1.63947. RTF 1 means real time. Pooling uses summed energies and duration-weighted measured time.

| Case | Profile | Waveform L2 | Tokens | Processing RTF | Startup-inclusive RTF | Waveform change |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p232_007 | baseline_current_board | 20.007% | 310/317 | 4.86711 | 7.50948 | unchanged |
| p232_007 | sorted | 31.772% | 303/317 | 4.87308 | 7.60455 | regressed |
| p232_007 | matrix | 16.104% | 307/317 | 6.45810 | 8.97241 | improved |
| p232_007 | matrix_up | 26.039% | 307/317 | 5.65559 | 8.30981 | regressed |
| p232_007 | matrix_down | 28.495% | 307/317 | 5.67323 | 8.27455 | regressed |

Regressions against the baseline:

- baseline_current_board: waveform: none; tokens: none; processing: none.
- sorted: waveform: p232_007; tokens: p232_007; processing: p232_007.
- matrix: waveform: none; tokens: p232_007; processing: p232_007.
- matrix_up: waveform: p232_007; tokens: p232_007; processing: p232_007.
- matrix_down: waveform: p232_007; tokens: p232_007; processing: p232_007.

Each FPGA run has one resident upload, one input upload, one START, one HALT, one output read, and zero CPU neural operations.

Startup-inclusive RTF includes runner artifact validation/loading and model upload. It excludes compilation, Python startup and final metrics serialization. Bin hashes are runner-recorded unless the JSON explicitly says the file was rehashed. No profile is selected automatically.
