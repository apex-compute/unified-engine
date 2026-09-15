# Bus pilot: token and decoder errors

The selected decoder-only candidate reduces total waveform error by 26.84% on this 23.68225 s bus case. Token-only error falls by 34.04%, and conditional decoder error by 19.29%.

| Variant | Matching tokens | Token-only error | Conditional decoder error | Total error | Error cosine | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | 1714/1895 | 29.6653% | 5.7347% | 28.0903% | -0.366653 | 4.79644 |
| Both LSTM stacks | 1725/1895 | 23.2538% | 4.9636% | 24.7075% | 0.202608 | 4.85071 |
| Decoder LSTM only (selected) | 1736/1895 | 19.5669% | 4.6284% | 20.5506% | 0.102492 | 4.82395 |

The official FP32 decoder reconstructed each recorded FPGA token sequence on CPU core 7, one thread. Token-only error measures the effect of the selected code IDs. Conditional decoder error includes FPGA embedding and decoder arithmetic for those IDs. The IDs differ between rows; this does not hold decoder inputs fixed across candidates.

The error vectors add. Their squared energies obey `E_total = E_token + E_decoder + 2⟨token_error, decoder_error⟩`:

- Before: 146.337624675 = 163.208116640 + 5.916385535 − 22.786877501.
- Both LSTM stacks: 113.214737047 = 100.283877802 + 4.409611534 + 8.521247711.
- Decoder LSTM only (selected): 78.323402322 = 71.004813256 + 3.905204374 + 3.413384692.

Baseline errors partly cancel; both candidates’ errors partly reinforce. This masks part of the component improvements. The conditional decoder percentage uses its own same-token reference energy; percentages cannot be added. No gain or timing adjustment was applied.

Both candidates use centered VQ scores and a compensated codebook. Compensated cell/tanh arithmetic and fused gates apply to the named LSTM stacks. All recurrent weights are BF16 in these pilots. The selected scope is used for the [full noisy-audio batch](decoder/); this single-case result does not remove the [short-clip regression](short_clip/README.md).

[Exact values, definitions, timing and artifact hashes](bus_error_decomposition.json)
