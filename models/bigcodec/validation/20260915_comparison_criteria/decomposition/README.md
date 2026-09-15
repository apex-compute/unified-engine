The previous serial BF16 baseline has **3.7070% pooled conditional decoder error** on eight long noisy files, while full-model waveform error remains **23.9405%**. Its separate short clip has **12.2874% conditional error**. No baseline file meets 10% end-to-end error. The later paired-gate results are reported [separately](../final/README.md).

| Case | Token-induced L2 | Conditional decoder L2 | Full-model L2 | Error cosine |
| --- | ---: | ---: | ---: | ---: |
| bus | 19.5669% | 4.6284% | 20.5506% | +0.102492 |
| cafe | 21.4300% | 2.7921% | 21.3314% | -0.100279 |
| office | 22.0969% | 3.5034% | 22.4779% | +0.033018 |
| psquare | 30.9258% | 3.6876% | 29.7715% | -0.370142 |
| bus_low_snr | 20.9516% | 3.9790% | 21.4418% | +0.033647 |
| cafe_low_snr | 33.1050% | 3.6673% | 32.8890% | -0.114128 |
| office_low_snr | 16.6838% | 3.4665% | 17.1790% | +0.043058 |
| psquare_low_snr | 22.2752% | 3.8054% | 22.6534% | +0.017588 |
| p232_007 | 14.1801% | 12.2874% | 20.0069% | +0.138658 |
| Eight long files pooled | 23.8749% | 3.7070% | 23.9405% | -0.058803 |

The short clip is excluded from the eight-file pool. Pooled errors use summed energies, not averages of percentages. These measurements compare the previous decoder-corrected BF16 FPGA implementation, before paired sigmoid gates, with the official FP32 model.

Let y be the official CPU output, z the official CPU decode of the FPGA tokens, and f the FPGA output. Token-induced error is ||z-y||₂/||y||₂; conditional decoder error is ||f-z||₂/||z||₂; full error is ||f-y||₂/||y||₂. Conditional error includes FPGA embedding, projection and decoder arithmetic.

For t=z-y and d=f-z, full error is t+d. Squared energies satisfy Efull=Etoken+Edecoder+2⟨t,d⟩. Negative error cosine indicates cancellation; positive cosine indicates reinforcement. Component percentages cannot be added. The results JSON includes energies, cross terms and centered error correlations.

The conditional criterion is useful for isolating decoder arithmetic; it does not demonstrate full-model accuracy below 10%. If downstream arithmetic exactly matched the official decoder at the existing FPGA token IDs, the eight-file pooled error would still be 23.8749%.

All WAV samples and token indices retain their original positions. No time, gain, polarity or sample-selection fitting was applied. Same-token CPU decodes were reused after verifying source, checkpoint, input-token and output hashes; no new neural or hardware inference was run for this report.

Reproduce all hashes, bindings and measurements with Python, NumPy and soundfile (no Torch, model checkpoint, or FPGA needed):

```bash
python models/bigcodec/validation/20260915_comparison_criteria/decomposition/reproduce.py
```

Run the command from the repository root. This directory adds only the eight same-token CPU WAVs in `audio/` and their decode records in `records/`. Existing official CPU and FPGA WAVs, token files, metrics, and all short-clip evidence are reused through hashed repository-relative paths. `manifest.json` inventories both new files and existing frozen references. The reproducer rejects paths outside the approved validation directories and does not depend on ignored evaluation caches.
