# DPDFNet2, 16 kHz

Run commands from `pcie_utils/`. This speech-enhancement port uses CEVA's pinned
[DPDFNet2 ONNX model](https://github.com/ceva-ip/DPDFNet). One 10 ms hop contains
160 audio samples, represented by a complex spectrum `[1, 1, 161, 2]` and
45,424 recurrent-state values. Audio STFT/iSTFT run on the host.

## Compile and run

```bash
python3 -m pip install -r requirements.txt
python3 models/dpdfnet/dpdfnet_compile.py --download
python3 models/dpdfnet/dpdfnet_run_from_bin.py \
  --bin models/dpdfnet/dpdfnet_bin/dpdfnet2-andromeda.bin \
  --device rk --dev xdma0 --input models/test_samples/apex.wav \
  --output /tmp/dpdfnet_16khz_fpga.wav

python3 models/dpdfnet/dpdfnet_run_cpu.py --download \
  --input models/test_samples/apex.wav --output /tmp/dpdfnet_16khz_cpu.wav
```

Compilation verifies the ONNX digest in [dpdfnet_config.json](dpdfnet_config.json)
and creates `dpdfnet_bin/dpdfnet2-andromeda.bin`. Add `--force` to rebuild;
`--output` chooses another bin path. The runner accepts `--bin` and loads only
the compiled image. Downloaded models and outputs remain local artifacts.

Default dense convolutions use IF4/IF8; other learned weights and activations
use BF16. For BF16 dense weights:

```bash
python3 models/dpdfnet/dpdfnet_compile.py --download --conv-precision bf16
python3 models/dpdfnet/dpdfnet_run_from_bin.py \
  --bin models/dpdfnet/dpdfnet_bin/dpdfnet2-bf16-andromeda.bin \
  --device rk --dev xdma0 --input models/test_samples/apex.wav \
  --output /tmp/dpdfnet_16khz_bf16.wav
```

Select the bin matching the compiled precision.
All neural operations execute on the FPGA. The model image is uploaded once;
each spectrum frame uses one input write, one START/HALT and one output read.
Recurrent state stays in device DRAM. The bins require RK AXI-256 with compatible
queue-CONFIG convolution hardware; the runner checks transport before upload.

WAV/FLAC inputs are downmixed and resampled to 16 kHz. Output is a mono float WAV
at the original sample rate and sample count. Four-hop delay compensation and
trailing silence recover the recording's end. No attenuation mixing,
normalization or clipping is applied by default. The CPU reference uses the
same audio framing. `.npy` spectrum input/output is also supported.

The runner writes `OUTPUT.metrics.json` with timings, deadline misses, hashes
and real-time factor; `--report` changes its path. `steady_audio_rtf` includes
host audio processing and excludes startup. RTF ≤ 1 means average real-time
throughput; check 10 ms deadline misses separately. `--cpu-core` optionally pins
the runner to an available CPU core.

## Noisy speech evaluation

Download the pinned VoiceBank-DEMAND subset and evaluate paired clean/noisy
recordings. [noisy_test_cases.json](noisy_test_cases.json) contains source URLs,
checksums, clip conditions and attribution; generated WAVs stay outside Git.

```bash
python3 -m pip install -r models/dpdfnet/requirements-eval.txt
python3 models/dpdfnet/dpdfnet_prepare_noisy.py --output-dir /tmp/dpdfnet-noisy
python3 models/dpdfnet/dpdfnet_evaluate_noisy.py \
  --manifest /tmp/dpdfnet-noisy/cases.json --output-dir /tmp/dpdfnet-eval \
  --backends cpu fpga --device rk --dev xdma0
```

Use `--limit N` on the preparer or evaluator for a subset. Evaluation reports
CPU/FPGA waveform agreement and clean-reference speech quality. Those are
separate measurements: agreement with CPU alone does not measure denoising.

## Tests and diagnostics

```bash
python3 -m unittest discover -s models/dpdfnet -p 'test_*.py'
```

For frame diagnostics, supply `.npy` input to `dpdfnet_run_from_bin.py` and use
`--trace-tail DIR`. The streaming runner reports end-to-end performance.
