# BigCodec on Andromeda: Benchmark Report

BigCodec is a neural audio codec operating at 16 kHz with a nominal token hop
of 12.5 ms. The Andromeda deployment compiles the complete encoder, vector
quantizer, decoder, recurrent state and activations into one resident artifact.


The FPGA path executes one complete utterance with one input upload, one program
kick, one HALT and one output read; `cpu_neural_ops=0` confirms that no neural
operation falls back to the host CPU.

The default benchmark input is
`test_samples/bigcodec_official/main/gt/1221-135767-0010.wav`. This official
BigCodec demo ground-truth file is clean mono 16 kHz audio containing 131,200
samples (8.2 seconds) and 657 token frames. 

`p232_007.wav` is not official traing/testing file, which is
a VoiceBank-DEMAND café-noise sample, requires 48 kHz to 16 kHz resampling, and covers 317 token frames. It is retained as a historical noisy-input stress test.
## Reproduce remotely


Compile the BF16 AXI-256 single-bin artifact:

```bash
mkdir -p perf_logs/bigcodec_<DEVICE_NAME> \
             perf_logs/bigcodec_cpu

python models/bigcodec/bigcodec_compile.py \
  --input test_samples/bigcodec_official/main/gt/1221-135767-0010.wav \
  --conv-precision bf16 \
  --lstm-precision bf16 \
  --lstm-cell-precision compensated \
  --lstm-tanh-precision compensated \
  --lstm-fused-gates \
  --lstm-math-scope decoder \
  --center-quantizer-scores \
  --compensated-codebook \
  --output models/bigcodec/bigcodec_bin/bigcodec-<DEVICE_NAME>-1221-135767-0010.bin \
  --force \
  2>&1 | tee perf_logs/bigcodec_<DEVICE_NAME>/compile_bf16.log
```

Generate the single-thread PyTorch FP32 CPU reference:

```bash
python models/bigcodec/bigcodec_run_cpu.py \
  --mode roundtrip \
  --input test_samples/bigcodec_official/main/gt/1221-135767-0010.wav \
  --output perf_logs/bigcodec_cpu/1221-135767-0010_cpu.wav \
  --tokens perf_logs/bigcodec_cpu/1221-135767-0010_cpu.tokens.npz \
  --metrics perf_logs/bigcodec_cpu/1221-135767-0010_cpu.metrics.json \
  --threads 1 \
  --cpu-core 7 \
  2>&1 | tee perf_logs/bigcodec_cpu/run_cpu.log

lscpu | tee perf_logs/bigcodec_cpu/lscpu.txt
```

Run the same artifact and input on the AXI-256 device. Replace every
`<DEVICE_NAME>` placeholder before executing. For the tested RK profile, the accepted value is `rk`
(not `rk_256`):

```bash
python models/bigcodec/bigcodec_run_from_bin.py \
  --bin models/bigcodec/bigcodec_bin/bigcodec-<DEVICE_NAME>-1221-135767-0010.bin \
  --input test_samples/bigcodec_official/main/gt/1221-135767-0010.wav \
  --output perf_logs/bigcodec_<DEVICE_NAME>/1221-135767-0010_fpga_bf16.wav \
  --tokens perf_logs/bigcodec_<DEVICE_NAME>/1221-135767-0010_fpga_bf16.tokens.npz \
  --report perf_logs/bigcodec_<DEVICE_NAME>/1221-135767-0010_fpga_bf16.metrics.json \
  --device <DEVICE_NAME> \
  --dev xdma0 \
  --cpu-core 6 \
  2>&1 | tee perf_logs/bigcodec_<DEVICE_NAME>/run_bf16.log
```

Compare the FPGA result against the CPU reference without gain, delay or
polarity fitting:

```bash
python models/bigcodec/bigcodec_compare.py \
  --reference perf_logs/bigcodec_cpu/1221-135767-0010_cpu.wav \
  --actual perf_logs/bigcodec_<DEVICE_NAME>/1221-135767-0010_fpga_bf16.wav \
  --reference-tokens perf_logs/bigcodec_cpu/1221-135767-0010_cpu.tokens.npz \
  --actual-tokens perf_logs/bigcodec_<DEVICE_NAME>/1221-135767-0010_fpga_bf16.tokens.npz \
  --report perf_logs/bigcodec_<DEVICE_NAME>/cpu_vs_fpga.json
```

### Official BigCodec demo input

The additional clean-speech test uses the first ground-truth sample from the
official BigCodec main-results demo, not a reconstructed codec output:

```text
https://aria-k-alethia.github.io/bigcodec-demo/audio/main/gt/8455-210777-0033.wav
```

Download and compile it with the same numerical profile:

```bash
mkdir -p test_samples/bigcodec_official \
             perf_logs/bigcodec_<DEVICE_NAME>/official_8455

curl -L --fail \
  --output test_samples/bigcodec_official/8455-210777-0033.wav \
  https://aria-k-alethia.github.io/bigcodec-demo/audio/main/gt/8455-210777-0033.wav

python models/bigcodec/bigcodec_compile.py \
  --input test_samples/bigcodec_official/8455-210777-0033.wav \
  --conv-precision bf16 \
  --lstm-precision bf16 \
  --lstm-cell-precision compensated \
  --lstm-tanh-precision compensated \
  --lstm-fused-gates \
  --lstm-math-scope decoder \
  --center-quantizer-scores \
  --compensated-codebook \
  --output /tmp/bigcodec-official-8455-<DEVICE_NAME>.bin \
  --force
```

Run the selected device and CPU, then compare them:

```bash
python models/bigcodec/bigcodec_run_from_bin.py \
  --bin /tmp/bigcodec-official-8455-<DEVICE_NAME>.bin \
  --input test_samples/bigcodec_official/8455-210777-0033.wav \
  --output perf_logs/bigcodec_<DEVICE_NAME>/official_8455/fpga.wav \
  --tokens perf_logs/bigcodec_<DEVICE_NAME>/official_8455/fpga.tokens.npz \
  --report perf_logs/bigcodec_<DEVICE_NAME>/official_8455/fpga.metrics.json \
  --device <DEVICE_NAME> --dev xdma0 --cpu-core 6

python models/bigcodec/bigcodec_run_cpu.py \
  --mode roundtrip \
  --input test_samples/bigcodec_official/8455-210777-0033.wav \
  --output perf_logs/bigcodec_<DEVICE_NAME>/official_8455/cpu.wav \
  --tokens perf_logs/bigcodec_<DEVICE_NAME>/official_8455/cpu.tokens.npz \
  --metrics perf_logs/bigcodec_<DEVICE_NAME>/official_8455/cpu.metrics.json \
  --threads 1 --cpu-core 7

python models/bigcodec/bigcodec_compare.py \
  --reference perf_logs/bigcodec_<DEVICE_NAME>/official_8455/cpu.wav \
  --actual perf_logs/bigcodec_<DEVICE_NAME>/official_8455/fpga.wav \
  --reference-tokens perf_logs/bigcodec_<DEVICE_NAME>/official_8455/cpu.tokens.npz \
  --actual-tokens perf_logs/bigcodec_<DEVICE_NAME>/official_8455/fpga.tokens.npz \
  --report perf_logs/bigcodec_<DEVICE_NAME>/official_8455/cpu_vs_fpga.json
```

## Performance metrics

### Core performance comparison

The CPU core time is FP32 PyTorch neural inference (`encode_s + decode_s`). The
FPGA core time is derived from its cycle counter. Model/artifact loading,
one-time model upload and audio file I/O are excluded from this table.

| Metric | Bittware AXI-256 | RK AXI-256 | CPU FP32 (1 thread, dust2) |
|---|---:|---:|---:|
| Audio duration | 8.200000 s | 8.200000 s | 8.200000 s |
| **Core execution time** | **45.979430 s** | **39.995159 s** | **16.197093 s** |
| Encoder time | Included | Included | 8.381910 s |
| Decoder time | Included | Included | 7.815183 s |
| Core RTF | 5.6072 | 4.8775 | 1.9753 |
| Audio processed per wall second | 0.1783× realtime | 0.2050× realtime | 0.5063× realtime |
| Finite output | Yes | Yes | Yes |
| Numerical validation | **Passes expected optimized BF16 profile** | **Passes expected optimized BF16 profile** | Reference |

RK (333MHz) executes the FPGA graph about
13.0% faster than Bittware (300MHz) for the same input and numerical profile.
### Host and deployment overhead

| Metric | Bittware AXI-256 | RK AXI-256 | CPU FP32 (1 thread, RK host) |
|---|---:|---:|---:|
| Artifact/model load | 7.032540 s | 7.227730 s | 1.191546 s |
| Model upload | 0.165044 s | 0.148396 s | N/A |
| Host-observed graph execution | 46.201160 s | 40.086737 s | 16.197093 s neural |
| Audio processing excluding load | 46.209383 s | 40.094369 s | 16.233279 s |
| Processing RTF | 5.6353 | 4.8896 | 1.9797 |
| Total elapsed | 59.403945 s | 54.009991 s | Not recorded on the same basis |
| Program kicks / HALTs | 1 / 1 | 1 / 1 | N/A |
| Intermediate host transfers | 0 uploads / 0 reads | 0 uploads / 0 reads | N/A |


### Correctness

Each FPGA result was compared against a fresh pinned official PyTorch FP32
reference generated from the identical input and checkpoint. No gain, delay,
polarity or alignment fitting is applied.

| Metric | Bittware AXI-256 optimized | RK AXI-256 |
|---|---:|---:|
| Samples / tokens | 131,200 / 657 | 131,200 / 657 |
| Relative L2 error | **0.121736** | **0.121736** |
| SNR | **18.291601 dB** | **18.291601 dB** |
| RMSE | **0.005018** | **0.005018** |
| Maximum absolute error | 0.164047 | 0.164047 |
| Cosine similarity | **0.992610** | **0.992610** |
| Token matches | **637 / 657 (96.956%)** | **637 / 657 (96.956%)** |
| Reference RMS | 0.041218 | 0.041218 |
| FPGA output RMS | 0.040513 | 0.040513 |
| Finite values | Yes | Yes |
| Status | **Validated optimized BF16 result** | **Validated optimized BF16 result** |

Both FPGA results have identical numerical metrics, perform the whole graph
with one START/HALT and no CPU neural operations, and are compared at the
original sample indices without gain, delay or polarity fitting.

### Optimization history

For historical context, the earlier `p232_007.wav` Bittware FPGA time was optimized from 69.275848s to 22.196159s. 

#### Historical `p232_007.wav` result (Bittware AXI-256 only)

This VoiceBank-DEMAND café-noise input is a 48 kHz source resampled to the
model's 16 kHz rate. The table records the final optimized Bittware result; RK
was not measured for this historical profile.

| Metric | Bittware AXI-256 |
|---|---:|
| Source samples / duration | 189,883 / 3.955896 s |
| Model-rate samples / tokens | 63,295 / 317 |
| FPGA core execution | 22.196159 s |
| Core RTF | 5.6109 |
| Host-observed graph execution | 22.299307 s |
| Audio processing excluding load | 22.443508 s |
| Processing RTF | 5.6734 |
| Artifact load | 3.709400 s |
| Model upload | 0.138396 s |
| Program kicks / HALTs | 1 / 1 |
| Intermediate host transfers | 0 uploads / 0 reads |
| Relative L2 error | 0.162277 |
| SNR | 15.794884 dB |
| RMSE | 0.011881 |
| Cosine similarity | 0.986763 |
| Token matches | 310 / 317 (97.79%) |
| Finite output | Yes |

The pre-optimization 69.275848-second FPGA result corresponds to a core RTF of
17.5124. The optimized 22.196159-second result is approximately 3.12x faster.

### All official demo input results

The official test set contains ten clean, mono 16 kHz LibriSpeech ground-truth
files published in the BigCodec main-results demo. Every Bittware measurement
below uses the same optimized BF16 profile: sorted encoder FIR, compensated
decoder cell/tanh and gates, centered quantizer scores, and compensated
codebook lookup. CPU figures are single-thread PyTorch FP32 references.

#### Per-input performance

| Official input | Duration | Bittware FPGA core | FPGA core RTF | CPU FP32 neural | CPU neural RTF |
|---|---:|---:|---:|---:|---:|
| `1221-135767-0010` | 8.200 s | 45.979430 s | 5.6072 | 17.189702 s | 2.0963 |
| `1284-1181-0008` | 6.080 s | 34.082713 s | 5.6057 | 12.363126 s | 2.0334 |
| `1580-141083-0013` | 4.320 s | 24.220800 s | 5.6067 | 7.196319 s | 1.6658 |
| `1995-1836-0009` | 6.710 s | 37.572594 s | 5.5995 | 13.305979 s | 1.9830 |
| `2830-3980-0051` | 6.440 s | 36.107025 s | 5.6067 | 12.687421 s | 1.9701 |
| `61-70970-0024` | 7.235 s | 40.518512 s | 5.6003 | 15.113322 s | 2.0889 |
| `672-122797-0069` | 5.010 s | 28.062210 s | 5.6012 | 8.364677 s | 1.6696 |
| `6829-68771-0003` | 4.015 s | 22.542911 s | 5.6147 | 6.825427 s | 1.7000 |
| `8455-210777-0033` | 7.510 s | 42.057325 s | 5.6002 | 15.023356 s | 2.0004 |
| `908-157963-0028` | 4.955 s | 27.787366 s | 5.6079 | 8.421052 s | 1.6995 |

#### Per-input correctness

| Official input | Token matches | Match rate | Relative L2 | SNR | Cosine similarity |
|---|---:|---:|---:|---:|---:|
| `1221-135767-0010` | 637 / 657 | **96.956%** | 0.121736 | 18.2916 dB | 0.992610 |
| `1284-1181-0008` | 431 / 487 | 88.501% | 0.177795 | 15.0016 dB | 0.984121 |
| `1580-141083-0013` | 321 / 346 | 92.775% | 0.112677 | 18.9633 dB | 0.993671 |
| `1995-1836-0009` | 485 / 537 | 90.317% | 0.185521 | 14.6322 dB | 0.982912 |
| `2830-3980-0051` | 488 / 516 | 94.574% | 0.240526 | 12.3768 dB | 0.970651 |
| `61-70970-0024` | 551 / 579 | 95.164% | 0.177676 | 15.0074 dB | 0.984232 |
| `672-122797-0069` | 379 / 401 | 94.514% | 0.217986 | 13.2314 dB | 0.975952 |
| `6829-68771-0003` | 298 / 322 | 92.547% | 0.196287 | 14.1422 dB | 0.981525 |
| `8455-210777-0033` | 547 / 601 | 91.015% | 0.182485 | 14.7755 dB | 0.983212 |
| `908-157963-0028` | 368 / 397 | 92.695% | 0.145350 | 16.7517 dB | 0.989407 |

#### Aggregate result

| Aggregate metric | Result |
|---|---:|
| Inputs | 10 |
| Total audio duration | 60.475 s |
| Total tokens | 4,843 |
| Token matches | **4,505 / 4,843 (93.021%)** |
| Total Bittware FPGA core time | 338.930886 s |
| Aggregate FPGA core RTF | 5.6045 |
| Total CPU FP32 neural time | 116.490382 s |
| Aggregate CPU neural RTF | 1.9263 |
| Mean per-file relative L2 | 0.175804 |
| Relative L2 range | 0.112677–0.240526 |
| Mean per-file SNR | 15.3174 dB |
| Mean per-file cosine similarity | 0.983829 |
| Token-match range | 88.501%–96.956% |

All outputs are finite. The aggregate token percentage is micro-averaged
(`total matches / total tokens`); waveform metrics are unweighted per-file
means. The selected default `1221-135767-0010.wav` has the highest token match
rate in this set. The previously verified official reconstruction cross-check
also established that the local PyTorch reference path agrees with the
developers' published output, so these residual errors characterize the
current BF16/approximate-arithmetic deployment rather than a reference-model
or preprocessing mismatch.

