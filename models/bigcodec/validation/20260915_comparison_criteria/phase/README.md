The baseline's magnitude-STFT difference accounts for **27.82%** of its squared complex-STFT error; the remaining **72.18%** is the phase-dependent term. This helps explain why sample-by-sample waveform error can be large despite closer magnitude spectra. It does not establish that the difference is inaudible.

For each complex STFT coefficient, the exact identity is:

```
|X - Y|² = (|X| - |Y|)² + 2(|X||Y| - Re(X conjugate(Y)))
```

| Existing implementation | Magnitude share of error energy | Phase-dependent share |
| --- | ---: | ---: |
| Serial baseline | 27.8151% | 72.1849% |
| Sorted encoder FIR | 25.3366% | 74.6634% |
| Matrix encoder FIR | 23.8424% | 76.1576% |

The calculation pools all eight frozen 16 kHz noisy cases, using a 1,024-sample periodic Hann window and 256-sample hop. These are shares of squared STFT error energy, not relative waveform errors. The independent short clip is excluded.

The strongest global cross-correlation peak within ±160 samples is at zero lag for every case and implementation. This check does not find a constant sample shift; it cannot exclude local timing changes or other structural errors. No gain, timing, polarity, or phase correction is applied to the reported audio.

`results.json` records exact settings, per-file values, source hashes, and identity checks. Recompute from the repository root:

```bash
python models/bigcodec/validation/20260915_comparison_criteria/phase/reproduce.py
```
