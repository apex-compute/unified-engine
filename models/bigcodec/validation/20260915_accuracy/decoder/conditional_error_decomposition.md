# Noisy-file waveform error decomposition

All eight complete noisy WAVs, in both recurrent precision modes. Saved FPGA tokens were decoded separately by the official FP32 CPU decoder on core 7, one thread. This is offline evaluation; the recorded FPGA inference used no CPU neural fallback.

| Recurrent weights | Run | Token-only error | Decoder error on identical tokens | Total waveform error | Tokens matching CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | baseline | 27.5358% | 5.7128% | 28.0572% | 12607/13845 |
| BF16 | selected | 23.8749% | 3.7070% | 23.9405% | 12796/13845 |
| IF8 | baseline | 26.7437% | 5.4185% | 27.0398% | 12617/13845 |
| IF8 | selected | 24.1727% | 4.3143% | 24.3495% | 12778/13845 |

Token-only error compares CPU decoding of FPGA tokens with the frozen official CPU reconstruction. Conditional decoder error compares the FPGA waveform with CPU decoding of those identical FPGA tokens, including embedding and decoder arithmetic. Before/after token sequences differ, so the conditional decoder inputs also differ.

Pooled values use summed error energies. Token-only and total errors use the frozen reference energy; conditional decoder error uses the corresponding same-token CPU energy. The JSON also includes decoder error normalized by the frozen reference energy.

For t = CPU(FPGA tokens) − CPU(CPU tokens) and d = FPGA − CPU(FPGA tokens), total error is t + d. Its energy is ‖t‖² + ‖d‖² + 2⟨t,d⟩. Negative cross terms cancel error; positive terms reinforce it. Component percentages cannot be added. No alignment, gain or polarity fitting was applied.

| Recurrent weights | Component | Cases improved | Cases regressed |
| --- | --- | ---: | ---: |
| BF16 | Total | 8/8 | 0/8 |
| BF16 | Token-only | 8/8 | 0/8 |
| BF16 | Conditional decoder | 8/8 | 0/8 |
| IF8 | Total | 7/8 | 1/8 |
| IF8 | Token-only | 7/8 | 1/8 |
| IF8 | Conditional decoder | 7/8 | 1/8 |

These metrics measure agreement with FP32 codec output, not noise suppression or perceptual quality. CPU diagnostic WAVs remain in the ignored evaluation directory; their exact hashes and input token hashes are recorded.

[Per-case energies, cross terms, counts and provenance](conditional_error_decomposition.json) · [Recorded FPGA benchmark](README.md)
