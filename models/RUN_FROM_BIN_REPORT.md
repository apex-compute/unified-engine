# Run-from-bin report: DPDFNet2 and BigCodec

Prepared **2026-09-16** from saved FPGA measurements on **2026-09-14–15**. No new hardware run was performed for this report.

**DPDFNet2 8 kHz IF8 meets average real-time throughput, with little margin. Its host still misses some 10 ms deadlines. DPDFNet2 16 kHz and BigCodec are slower than real time.** All listed FPGA implementations execute neural inference from a compiled single bin.

Platform: **Kintex UltraScale+ KU5P**, RK **AXI-256**, Italy host, measured FPGA clock **333.25 MHz**. Host CPU: Intel Core Ultra 9 285K; FPGA runner pinned to core 6. Saved CPU references use one thread on core 7. Individual measurement sets use the FPGA builds identified below.

**RTF = processing seconds / audio seconds.** Below 1 means faster than real time; speed is `1 / RTF`. Processing RTF excludes one-time bin loading, device initialization and model upload. It includes the host audio-processing work. A low average RTF does not guarantee every streaming deadline.

## DPDFNet2: native 8 kHz and 16 kHz

All four variants processed the same two noisy bus/café recordings: **42.3448125 seconds, 4,247 hops**, including end flush. Measured **September 14**, FPGA build **`0x40519e0a`**. The 8 kHz implementation uses the official native 8 kHz checkpoint.

| Model / dense convolution weights | FPGA ms/hop | Host ms/hop | Processing seconds | RTF | Speed | Waveform error vs own FP32 CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 kHz / BF16 | 9.591 | 10.398 | 44.228 | **1.0445** | 0.957× | 2.728% |
| 8 kHz / IF8 | 9.070 | 9.865 | 41.965 | **0.9910** | 1.009× | 2.675% |
| 16 kHz / BF16 | 14.085 | 14.936 | 63.481 | **1.4992** | 0.667× | 3.189% |
| 16 kHz / IF4/IF8 | 13.447 | 14.291 | 60.740 | **1.4344** | 0.697× | 5.035% |

Host frame timing covers input packing/transfer, execution and output retrieval. STFT/iSTFT and WAV writing are included separately in the processing RTF. Every measured output was finite.

| Variant | Host hops exceeding 10 ms | FPGA executions exceeding 10 ms |
| --- | ---: | ---: |
| 8 kHz BF16 | 4,100 / 4,247 | 0 / 4,247 |
| 8 kHz IF8 | 838 / 4,247 | 0 / 4,247 |
| 16 kHz BF16 | 4,247 / 4,247 | 4,247 / 4,247 |
| 16 kHz IF4/IF8 | 4,247 / 4,247 | 4,247 / 4,247 |

Both 8 kHz variants finish FPGA computation within 10 ms, but host overhead removes the BF16 throughput margin. Both models also have **40 ms algorithmic delay**, separate from processing time.

Precision names describe dense convolution weights. At 8 kHz, all 15 dense kernels use BF16 or IF8. The 16 kHz quantized variant uses **12 IF4 and 3 IF8** kernels. Other learned weights and stored activations remain BF16. IF4/IF8 are the engine's encoded weight formats; these labels do not establish a pure INT4/INT8 network. BF16 uses device im2col/matrix instructions, while quantized variants use native CONV, so the timing comparison includes implementation differences. Internal BF19/BF20 arithmetic is unchanged.

Waveform error is pooled relative L2, `sqrt(sum((FPGA - CPU)^2) / sum(CPU^2))`, against each rate's own FP32 ONNX reference. It uses every output sample after fixed model-delay compensation, with no fitted gain/alignment or trimming. It measures numerical agreement, not denoising quality. References were reused and hash-verified; CPU timing was not remeasured for this four-variant comparison. These long-file aggregates do not imply every clip has the same error: earlier 16 kHz short-clip tests reached 11.218% and 24.768%.

[Detailed DPDFNet report](dpdfnet/RK256_PRECISION_BENCHMARK.md) · [Aggregate data](dpdfnet/validation/20260914_bf16/summary.json) · [Run records](dpdfnet/validation/20260914_bf16/results.json)

## BigCodec: native 16 kHz

BigCodec compresses and reconstructs audio; it is not a noise-suppression model. Both rows below use the same **eight noisy recordings, 173.0098125 seconds total**, with individual files around 20–24 seconds. They are separate implementation revisions, not a controlled change of precision alone.

| Implementation | Processing seconds | Processing RTF | Speed | RTF including startup | Full waveform error vs FP32 CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latest BF16, compensated decoder, counted LSTM loops | 844.121 | **4.8790** | 0.205× | 6.3353 | **23.909%** |
| Earlier IF8 recurrent weights, BF16 convolutions/activations | 758.245 | **4.3827** | 0.228× | 6.6682 | **24.350%** |

The latest BF16 results were recorded September 15 on **`0x90f1f464`**. The earlier IF8 results use **`0x40519e0a`** and predate the latest paired decoder arithmetic and counted loops. That IF8 speed must not be presented as a fresh benchmark of the current compiler. Latest BF16 native FPGA execution RTF is **4.8683**. At processing RTF 4.879, 20 seconds of audio needs about **97.6 seconds**, excluding startup.

Startup-inclusive timing adds artifact loading/validation, device initialization and resident upload. It excludes Python process startup and final report serialization.

The BF16 path stores convolution/recurrent weights and activations in BF16; selected decoder state and nonlinear calculations retain a high/low BF16 pair for added precision. The IF8 variant quantizes recurrent weights in both LSTMs; convolution and LSTM input-projection weights remain BF16. Neither variant is an all-FP32 implementation.

For latest BF16, CPU decoding of the **exact FPGA tokens** gives **3.091% pooled decoder-only waveform error**, with every long clip below 10%. Full encoder-plus-decoder error remains **23.909%**, and every long clip remains above 10%. CPU and FPGA token IDs match **12,796 / 13,845 (92.423%)**. The decoder-only result isolates arithmetic error; it does not replace the full-model comparison.

Mean agreement scores against the official CPU reconstruction are **PESQ-WB 4.0254** and **STOI 0.9790**. Against the same original noisy input, FPGA-minus-CPU score gaps are **−0.0110 PESQ-WB** and **−0.00188 STOI**. These are codec-fidelity comparisons, not clean-speech denoising scores or percentages of words understood.

[Latest BF16 results and all output WAVs](bigcodec/validation/20260915_comparison_criteria/final/README.md) · [Latest data](bigcodec/validation/20260915_comparison_criteria/final/results.json) · [Earlier IF8/BF16 results](bigcodec/validation/20260915_accuracy/decoder/README.md)

## CPU comparison on the broader noisy corpus

These runs use the same **eight files / 173.0098125 seconds**. The DPDFNet rows are earlier measurements on **`0xdf0749de`**; their dataset differs from the two-file precision table above. CPU baselines use each model's FP32 reference, one thread. DPDFNet uses ONNX Runtime; BigCodec uses its official PyTorch implementation.

| Model / implementation | FPGA processing RTF | FP32 CPU processing RTF | FPGA waveform error vs CPU |
| --- | ---: | ---: | ---: |
| DPDFNet2 8 kHz IF8 | 0.98985 | 0.06740 | 1.683% |
| DPDFNet2 16 kHz IF4/IF8 | 1.43636 | 0.07043 | 4.053% |
| BigCodec latest BF16 | 4.87904 | 2.32369 | 23.909% |

All three FPGA implementations are slower than their measured single-thread CPU references. The tasks differ: DPDFNet enhances speech; BigCodec reconstructs compressed audio.

[DPDFNet eight-file measurements](dpdfnet/validation/20260914_noisy20s/summary.json) · [BigCodec CPU/FPGA measurements](bigcodec/validation/20260915_comparison_criteria/final/results.json)

## What “single bin” executes

Each deployment bin contains its complete neural program and packed parameter/data sections. The runner uploads the resident image to DRAM once; program and parameters are not separate per-layer host uploads. All neural operators execute on the FPGA, with no intermediate host tensor transfers or CPU neural fallback.

| Property | DPDFNet2 8 kHz | DPDFNet2 16 kHz | BigCodec |
| --- | --- | --- | --- |
| Neural execution unit | One 10 ms hop | One 10 ms hop | One complete compiled-length WAV |
| New audio samples | 80 | 160 | Entire input file at native 16 kHz |
| Host audio processing | STFT / iSTFT | STFT / iSTFT | Resampling, padding, WAV I/O |
| Input/output at DRAM boundary | BF16 spectrum / enhanced spectrum | BF16 spectrum / enhanced spectrum | Audio tensor / reconstructed audio plus tokens |
| Input write, START, HALT, output read | One of each per hop | One of each per hop | One of each per file |
| Input / output bytes per hop | 10,368 / 10,368 | 20,608 / 20,608 | Depends on compiled file length |
| Neural recurrent state | Retained in DRAM between hops | Retained in DRAM between hops | Managed inside whole-file execution |

DPDFNet's STFT uses an overlapping 20 ms window advanced by 10 ms. Its DRAM input is a spectrum, not raw microphone PCM. These tests read recorded WAVs, rather than live microphone capture. Output files preserve the input WAV rate and sample count, so an 8 kHz model can produce a 16 kHz WAV after resampling.

BigCodec's **200-sample / 12.5 ms token interval is not a streaming execution interval**. The validated implementation does not implement the requested 10 ms START/HALT streaming contract; it implements one START/HALT for the entire file. Its bin must match the padded input length.

[Observed DPDFNet transfers](dpdfnet/validation/20260914_bf16/frame_contract.json) · [BigCodec execution receipts](bigcodec/validation/20260915_comparison_criteria/final/results.json)

## Instruction and parameter sections

Sizes below are **bytes**. Program and parameter/data are sections of one bin. Resident size is their sum; the serialized file also contains metadata. Parameter/data includes constants, initial state, alignment and replicated weight streams, so its size is not just learned-weight count. Runtime tensor workspace is additional.

| Deployment | Instructions | Program bytes | Parameter/data bytes | Resident bytes | Serialized bin bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPDFNet2 8 kHz BF16 | 28,058 | 897,856 | 14,538,368 | 15,436,224 | 15,566,040 |
| DPDFNet2 8 kHz IF8 | 27,118 | 867,776 | 22,726,272 | 23,594,048 | 23,721,269 |
| DPDFNet2 16 kHz BF16 | 41,090 | 1,314,880 | 13,220,224 | 14,535,104 | 14,658,165 |
| DPDFNet2 16 kHz IF4/IF8 | 40,264 | 1,288,448 | 16,575,360 | 17,863,808 | 17,984,019 |
| BigCodec BF16, default 3.956 s clip | 2,200,886 | 70,428,352 | 331,267,968 | 401,696,320 | 401,738,031 |
| BigCodec BF16, 23.682 s bus clip | 12,990,548 | 415,697,536 | 331,267,968 | 746,965,504 | 747,007,471 |

Across the eight long clips, latest BigCodec BF16 programs range from **359,998,592 to 415,697,536 bytes**, with constant **331,267,968 parameter/data bytes** and **691,266,560–746,965,504 resident bytes**. Earlier IF8 programs use **572,244,160–660,894,016 bytes**, parameters **294,698,880 bytes**, and resident images **866,943,040–955,592,896 bytes**. Their larger programs reflect the older unrolled implementation.

DPDFNet bin filenames are listed in its [precision report](dpdfnet/RK256_PRECISION_BENCHMARK.md#single-bin-execution-and-sizes). BigCodec's default `bigcodec-andromeda.bin` is for the short `p232_007.wav` example; the long-file measurements use separate bins under `models/bigcodec/bigcodec_bin/accuracy_next_20260915/`. Bins are generated local artifacts; source, measurements and sample WAVs are tracked in Git.

## Listen to the same noisy input across models

The following files all correspond to the **20.879-second bus low-SNR recording**. Each output has the same source sample rate/count. The DPDFNet outputs are speech enhancement; the BigCodec output is codec reconstruction.

| Audio | WAV |
| --- | --- |
| Noisy input | [bus_low_snr_noisy.wav](dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav) |
| DPDFNet2 8 kHz BF16 | [FPGA output](dpdfnet/validation/20260914_bf16/fpga8k_bf16/bus_low_snr_fpga_8khz_bf16.wav) |
| DPDFNet2 8 kHz IF8 | [FPGA output](dpdfnet/validation/20260914_bf16/fpga8k_quantized/bus_low_snr_fpga_8khz_quantized.wav) |
| DPDFNet2 16 kHz BF16 | [FPGA output](dpdfnet/validation/20260914_bf16/fpga16k_bf16/bus_low_snr_fpga_16khz_bf16.wav) |
| DPDFNet2 16 kHz IF4/IF8 | [FPGA output](dpdfnet/validation/20260914_bf16/fpga16k_quantized/bus_low_snr_fpga_16khz_quantized.wav) |
| BigCodec latest BF16 | [FPGA output](bigcodec/validation/20260915_comparison_criteria/final/fpga/bus_low_snr.wav) |
| BigCodec official FP32 CPU | [CPU reconstruction](bigcodec/validation/20260914_noisy20s/cpu/bus_low_snr.wav) |

## Run the existing bins on Italy

Run sequentially from the repository root with the compatible RK AXI-256 FPGA image already loaded. These commands perform WAV-to-WAV inference from existing bins; they do not compile or flash. Italy's existing environment is `/home/hunlu/my_torch_env/bin/python`; dependency setup is documented in the [DPDFNet report](dpdfnet/RK256_PRECISION_BENCHMARK.md#reproduce) and [BigCodec README](bigcodec/README.md).

The DPDFNet entry points select and validate their matching bins. The shared hardware lock is held for each run:

```bash
run_dpdfnet() (
  flock -n 9 || exit 1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hunlu/my_torch_env/bin/python "$1" \
    --device rk --dev xdma0 --cpu-core 6 --timeout 10 \
    --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav \
    --output "$2"
) 9</tmp/pcie_ci_hw_italy.lock

run_dpdfnet models/dpdfnet8khz/dpdfnet8khz_bf16_weights_test.py /tmp/bus_8k_bf16.wav
run_dpdfnet models/dpdfnet8khz/dpdfnet8khz_if8_weights_test.py /tmp/bus_8k_if8.wav
run_dpdfnet models/dpdfnet/dpdfnet16khz_bf16_weights_test.py /tmp/bus_16k_bf16.wav
run_dpdfnet models/dpdfnet/dpdfnet16khz_if4_if8_weights_test.py /tmp/bus_16k_if4_if8.wav
```

BigCodec acquires the hardware lock internally. Use the matching long-file bin for the same recording:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hunlu/my_torch_env/bin/python \
  models/bigcodec/bigcodec_run_from_bin.py \
  --bin models/bigcodec/bigcodec_bin/accuracy_next_20260915/bus_low_snr_paired_production.bin \
  --input models/dpdfnet/validation/20260914_noisy20s/noisy/bus_low_snr_noisy.wav \
  --output /tmp/bus_bigcodec_bf16.wav --cpu-core 6
```
