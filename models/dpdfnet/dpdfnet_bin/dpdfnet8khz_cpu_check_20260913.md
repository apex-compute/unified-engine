# Native 8 kHz CPU check, 2026-09-13

The official `dpdfnet2_8khz` model was downloaded and run on Italy using
`dpdfnet==0.6.0` and ONNX Runtime CPUExecutionProvider. Its SHA-256 is
`6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d`.
The model uses 8,000 Hz, a 160-sample window/FFT, an 80-sample (10 ms) hop,
81 spectrum bins, 37,860 recurrent-state elements and 492 ONNX nodes.

`test_samples/p232_007.wav` was resampled from 48 to 8 kHz with SciPy
`resample_poly(audio, 1, 6)`, producing the included FLOAT noisy WAV. The
upstream CLI produced the included PCM16 enhanced WAV; both are mono,
31,648 samples (3.956 seconds) and finite. The original sample attribution
and license apply; this checkpoint adds resampled and enhanced derivatives.
See [source attribution](../../../test_samples/p232_007.README.md).

```bash
source /home/hunlu/my_torch_env/bin/activate
python -m pip install dpdfnet==0.6.0
dpdfnet download dpdfnet2_8khz
dpdfnet enhance \
  models/dpdfnet/dpdfnet_bin/p232_007_noisy_8khz.wav \
  models/dpdfnet/dpdfnet_bin/p232_007_enhanced_cpu_8khz.wav \
  --model dpdfnet2_8khz
```

- [Noisy 8 kHz input](p232_007_noisy_8khz.wav)
- [Enhanced 8 kHz CPU output](p232_007_enhanced_cpu_8khz.wav)
- [Actual CLI log](p232_007_cpu_8khz.log)
- [Validation and hashes](dpdfnet8khz_cpu_validation_20260913.json)

This is a CPU execution/format check, not a quality or timing benchmark.
The existing 16 kHz FPGA bin was not used for this check. Native 8 kHz FPGA
implementation begins separately under `models/dpdfnet8khz/`.
