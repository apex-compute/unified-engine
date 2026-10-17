# DPDFNet2, native 8 kHz

Run commands from `pcie_utils/`. This port uses CEVA's official `dpdfnet2_8khz`
ONNX graph. Its 160-sample window and 80-sample hop represent 20 ms and 10 ms
at 8 kHz. The model uses 81 frequency bins and 37,860 recurrent-state values.
The [16-kHz port](../dpdfnet/README.md) provides the shared implementation.

```bash
python3 -m pip install -r requirements.txt
python3 models/dpdfnet8khz/dpdfnet8khz_compile.py --download
python3 models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --bin models/dpdfnet8khz/dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin \
  --device rk --dev xdma0 --input models/test_samples/apex.wav \
  --output /tmp/dpdfnet_8khz_fpga.wav

python3 models/dpdfnet8khz/dpdfnet8khz_run_cpu.py --download \
  --input models/test_samples/apex.wav --output /tmp/dpdfnet_8khz_cpu.wav
```

Compilation verifies the ONNX digest in
[dpdfnet8khz_config.json](dpdfnet8khz_config.json) and writes
`dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin`. Use `--force` to rebuild, `--output`
for another destination, or `--baseline` to disable optimized lowering.
The runner accepts `--bin`; checkpoints, bins and outputs remain local.

Default dense weights use IF8; depthwise/recurrent weights and activations use
BF16. To compile and run BF16 dense weights:

```bash
python3 models/dpdfnet8khz/dpdfnet8khz_compile.py --download --conv-precision bf16
python3 models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --bin models/dpdfnet8khz/dpdfnet8khz_bin/dpdfnet2_8khz-bf16-andromeda.bin \
  --device rk --dev xdma0 --input models/test_samples/apex.wav \
  --output /tmp/dpdfnet_8khz_bf16.wav
```

Select the bin matching the compiled precision. The runner requires a compatible
RK AXI-256 queue-CONFIG image. It uploads parameters/program once, then performs one input
write, one START/HALT and one output read per 80-sample hop. All 492 neural graph
operations and the recurrent-state commit execute on the FPGA. STFT/iSTFT run
on the host.

Input audio is downmixed and resampled to 8 kHz. Output preserves the original
sample rate and sample count in a mono float WAV. Four-hop model delay is
compensated after appending silence. Optional `--attn-limit-db` on the CPU
runner limits suppression by blending the aligned noisy spectrum.

`OUTPUT.metrics.json` records per-frame FPGA/host times, percentiles, deadline
misses, input/output/bin hashes and detected FPGA build. `--report` selects
another path. `steady_audio_rtf` includes host audio processing and excludes
startup. RTF ≤ 1 indicates average real-time throughput; individual frames may
still miss the 10 ms deadline. The 40 ms algorithmic delay is separate from
processing time. `--cpu-core` optionally selects an available host CPU core.

Prepare reference cases, including silence and speech/silence transitions:

```bash
python3 models/dpdfnet8khz/dpdfnet8khz_prepare_validation.py --download \
  --input models/test_samples/apex.wav --output /tmp/dpdfnet8khz-cases
(
  for input in /tmp/dpdfnet8khz-cases/*_input.npy; do
    case_base="${input%_input.npy}"
    python3 models/dpdfnet8khz/dpdfnet8khz_run_from_bin.py \
  --bin models/dpdfnet8khz/dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin \
      --device rk --dev xdma0 --input "$input" --output "${case_base}_fpga.npy" \
      > "${case_base}_fpga.log" 2>&1 || exit 1
  done
)
python3 models/dpdfnet8khz/dpdfnet8khz_compare.py \
  --reference-manifest /tmp/dpdfnet8khz-cases/cpu_reference_manifest.json \
  --fpga-dir /tmp/dpdfnet8khz-cases --output /tmp/dpdfnet8khz-comparison.json
```

Repeat `--input` to add recordings. The comparison checks hashes, numerical
agreement, silence and measured deadlines. Its coarse spectrum-error threshold
is not a speech-quality criterion. See the [16-kHz guide](../dpdfnet/README.md)
for pinned noisy recordings and paired clean/noisy evaluation.

```bash
python3 -m unittest discover -s models/dpdfnet8khz -p 'test_*.py'
```
