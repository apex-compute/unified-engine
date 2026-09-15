# Decoder correction and comparison criteria

The decoder discarded small sigmoid-gate terms when storing each gate as one BF16 value. Retaining a BF16 high/low pair through the LSTM cell and hidden products reduces the separate short clip's full waveform error **20.01% → 16.23%**, and decoder error with identical tokens **12.29% → 6.12%**. The encoder, checkpoint and codebook choices are unchanged.

**Full-model waveform error is still above 10%.** The [final FPGA results](final/README.md) compare the corrected implementation on all eight noisy files and the separate short clip, including WAVs, runtime, bin sizes and exact error values. Each run uses one resident bin, one input upload, one START/HALT and one output read, with all neural operations on the FPGA.

| Eight noisy files, 173.01 seconds | Previous | Corrected |
| --- | ---: | ---: |
| Pooled full waveform L2 | 23.9405% | 23.9094% |
| Pooled same-token decoder L2 | 3.7070% | 3.0913% |
| Mean PESQ-WB against CPU output | 4.02199 | 4.02539 |
| Mean STOI against CPU output | 0.979003 | 0.979017 |
| Processing RTF | 4.824 | 4.879 |
| RTF including runner startup | 7.177 | 6.335 |

All nine same-token decoder errors are below 10%: **2.85–3.31%** on long clips and **6.12%** on the short clip. Full error improves on five long files and regresses on `psquare`, `cafe_low_snr` and `psquare_low_snr`; same-token error has a small `cafe` regression. Perceptual-score changes are small. The prepared default bin uses this correction; [hashes, instruction/parameter sizes and preserved previous bin](deployment.json) are recorded. No new user-selectable profile was added.

Use three complementary checks:

| Check | Reference | What it measures |
| --- | --- | --- |
| Full waveform relative L2 | Official CPU encode/decode of the same input | Complete numerical agreement, including changed token IDs |
| Conditional decoder relative L2 | Official CPU decoding the FPGA's exact tokens | Embedding/projection/decoder arithmetic without encoder token differences |
| PESQ-WB and STOI, plus score differences from CPU | CPU reconstruction, and separately the same original input for both implementations | Speech quality agreement and the FPGA's additional effect on codec quality |

PESQ and STOI are also used in the [BigCodec paper](https://arxiv.org/html/2409.05377v1#S4.SS1). Their scores are not error percentages, and our noisy-input tests cannot be compared directly with the paper's clean LibriSpeech scores. With a noisy or codec-reconstructed reference, these are agreement measurements rather than validated clean-reference intelligibility scores. Raw waveform L2 stays in every comparison. No external gain, delay or polarity fitting is used.

The [baseline decomposition](decomposition/README.md) shows why a more accurate decoder alone cannot eliminate the full error: exact CPU decoding of the baseline FPGA tokens still differs **23.87%** from the complete CPU round trip across the eight files. The [phase analysis](phase/README.md) attributes **72.18% of baseline squared complex-STFT error energy** to the phase-dependent term, with zero best global lag in every case. This does not establish inaudibility or rule out every structural defect.

[Native gate controls](paired_gates/README.md) verify the numerical correction; the encoder variant was rejected because token agreement worsened. [Counted-loop controls](loop/README.md) verify bit-identical recurrence with a smaller instruction program. [Software validation](software_tests.json): **181 tests pass**.

To compare new WAVs:

```bash
python -m pip install -r models/bigcodec/requirements-quality.txt
python models/bigcodec/bigcodec_audio_quality.py \
  --reference reconstructed_cpu.wav --actual reconstructed_fpga.wav \
  --report quality.json
```

The [earlier three-implementation quality audit](perceptual/comparison.md) retains all previous results, including regressions. It is separate from the final corrected implementation.
