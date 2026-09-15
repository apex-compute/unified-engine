Counted FPGA timestep loops preserve the LSTM output exactly while reducing the resident program. Each layer initializes its state, executes timestep zero, then advances input/output DRAM addresses inside one device loop. Cell state and weights stay resident; there is no additional host execution boundary.

| Isolated decoder, 317 timesteps | Unrolled program | Counted program | Native output mismatches |
| --- | ---: | ---: | ---: |
| Previous gate arithmetic | 38,422,272 B | 1,257,280 B | 0 |
| Paired sigmoid gates | 93,281,024 B | 1,603,392 B | 0 |

The counted runs took 1.9503 s and 2.1676 s, respectively, on image `0x90f1f464`. Each used one START/HALT; inputs and guards were unchanged. The small timing difference from unrolled execution is recorded in the paired-gate receipts. The saved unrolled reference tensors are in [paired_gates/tensors](../paired_gates/tensors/).

The tests expand loop addresses and arithmetic calls against the unrolled implementation, verify the bias pointer's zero descriptor deltas, and check absolute jumps and register release. The production sigmoid primitive and complete timestep also match the original native-validated prototype's instruction bytes exactly. [Native and capture receipts](manifest.json).

The unchanged 23.68225-second bus deployment bin was also rerun on `0x90f1f464`. All waveform sample bits and tokens match the earlier `0x40519e0a` result; processing RTF is 4.82393. [Current-image control](bus_baseline_image_parity.json). This is a control of the previous implementation, separate from the new paired-gate result.

`loop_lstm.py` and `loop_native.py` here are exact historical diagnostic snapshots, with their original imports and paths. The maintained implementation and executable tests are `models/bigcodec/bigcodec_lstm_loop.py` and `models/bigcodec/test_bigcodec_lstm_loop.py`.
