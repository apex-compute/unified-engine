# Complementary codec metrics

Waveform L2 remains the numerical agreement measure. PESQ/STOI and magnitude spectra provide additional views; their scores do not certify waveform error below10%. No external gain, delay or polarity fitting is used.

## eight_noisy

Original noisy codec input; all eight cases.

Reference: **official_cpu**.

| Output | Pooled waveform L2 | Mean PESQ-WB | Mean PESQ-NB | Mean STOI | Pooled MR magnitude convergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| official_cpu | 0.00000% | 4.64389 | 4.54864 | 1.000000 | 0.00000% |
| serial_baseline | 23.94055% | 4.02199 | 4.19402 | 0.979003 | 12.50024% |
| sorted_encoder | 22.63993% | 4.02530 | 4.21493 | 0.977658 | 11.29958% |
| matrix_encoder | 23.84679% | 4.04522 | 4.20663 | 0.976811 | 11.48963% |

Reference: **input**.

| Output | Pooled waveform L2 | Mean PESQ-WB | Mean PESQ-NB | Mean STOI | Pooled MR magnitude convergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| input | 0.00000% | 4.64389 | 4.54864 | 1.000000 | 0.00000% |
| official_cpu | 84.68387% | 1.81483 | 2.65375 | 0.819734 | 42.15995% |
| serial_baseline | 82.83272% | 1.80216 | 2.63741 | 0.817776 | 42.42549% |
| sorted_encoder | 85.84908% | 1.80264 | 2.64292 | 0.818512 | 42.28358% |
| matrix_encoder | 82.86190% | 1.79906 | 2.63532 | 0.817484 | 42.25847% |

## short_clip

Separate speech clip; excluded from eight-case aggregation.

Reference: **official_cpu**.

| Output | Pooled waveform L2 | Mean PESQ-WB | Mean PESQ-NB | Mean STOI | Pooled MR magnitude convergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| official_cpu | 0.00000% | 4.64389 | 4.54864 | 1.000000 | 0.00000% |
| serial_baseline | 20.00685% | 4.52645 | 4.48788 | 0.997917 | 5.38379% |
| sorted_encoder | 31.77150% | 4.35687 | 4.35448 | 0.995977 | 6.76979% |
| matrix_encoder | 16.10446% | 4.46704 | 4.42978 | 0.995202 | 5.91867% |

Reference: **input**.

| Output | Pooled waveform L2 | Mean PESQ-WB | Mean PESQ-NB | Mean STOI | Pooled MR magnitude convergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| input | 0.00000% | 4.64389 | 4.54864 | 1.000000 | 0.00000% |
| official_cpu | 122.17770% | 1.89262 | 2.41246 | 0.864030 | 32.53500% |
| serial_baseline | 114.79802% | 1.89206 | 2.39809 | 0.863755 | 32.41262% |
| sorted_encoder | 109.07196% | 1.90924 | 2.41989 | 0.863617 | 32.61379% |
| matrix_encoder | 124.77129% | 1.90304 | 2.41065 | 0.862835 | 32.57054% |

Eight-case data are16kHz. The independent short clip is48kHz: raw waveform L2 stays at48kHz, while each signal is identically resampled to16kHz for the speech/spectral metrics. Short results are never pooled into the eight-case results.

PESQ/STOI are arithmetic means across files, not percentages of words understood. PESQ retains its internal level/time processing; STOI retains silence removal, resampling and envelope normalization. Magnitude convergence ignores phase and has no equivalence to the10% waveform target. Detailed settings, versions, individual results and source hashes are in the JSON.

The matrix psquare waveform error is37.0452%, while PESQ-WB is4.1834 and STOI is0.98585; these measures capture different properties. The CPU codec also changes the original noisy input substantially. Agreement with its output does not establish noise suppression.

[Primary sources and interpretation](sources.md), [eight-case results](eight_noisy_results.json), [separate short results](short_clip_results.json), [input manifest](manifest.json).

Reproduce without model weights, deployment bins or an FPGA:

```bash
pip install -r models/bigcodec/requirements-quality.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python models/bigcodec/validation/20260915_comparison_criteria/perceptual/generate_report.py \
  --output-dir /tmp/bigcodec-quality-recomputed
```
