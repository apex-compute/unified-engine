# BigCodec on Italy

The complete encoder, codebook lookup and decoder execute on the RK AXI256 FPGA from one resident bin. This is an **experimental port**: reconstructed audio differs from the official FP32 model, and execution is slower than real time.

Hardware: Kintex UltraScale+ KU5P, version `0xdf0749de`, AXI256, reported 333.25 MHz. Host: Intel Core Ultra 9 285K; FPGA runner on core 6, FP32 CPU reference on core 7, one math thread each. Checkpoint SHA256: `1fba3806e87cc01c1a65bea22fa1becefbbf46881e4219593c4d9f3cf56206b9`.

## Full speech recording

The 3.955896-second input has 189,883 samples at 48 kHz. The model processes 63,295 samples at its native 16 kHz rate, padded to 63,400 samples, producing 317 tokens. Output WAVs return to the original rate and sample count.

| Backend | File processing | RTF | Speed relative to real time | Tokens matching FP32 | Waveform error vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU FP32 | 6.455 s | 1.632 | 0.613× | 317/317 | Reference |
| FPGA IF8 | 70.996 s | 17.947 | 0.0557× | 297/317 | 22.50% |
| FPGA BF16 | 60.614 s | 15.322 | 0.0653× | 301/317 | 19.29% |

RTF is file-processing time divided by audio duration; real time requires RTF ≤ 1. Timings include audio preprocessing, inference, input/output DMA for FPGA, and output restoration/write. They exclude checkpoint/bin loading, model upload and compilation. CPU neural inference alone took 6.302 s. FPGA execution including input/output DMA took 70.867 s for IF8 and 60.484 s for BF16; each one-time model upload took approximately 0.187 s.

- [p232_007_input.wav](p232_007_input.wav)
- [p232_007_cpu.wav](p232_007_cpu.wav)
- [p232_007_fpga_if8.wav](p232_007_fpga_if8.wav)
- [p232_007_fpga_bf16.wav](p232_007_fpga_bf16.wav)
- [p232_007_cpu.tokens.npz](p232_007_cpu.tokens.npz)
- [p232_007_fpga_if8.tokens.npz](p232_007_fpga_if8.tokens.npz)
- [p232_007_fpga_bf16.tokens.npz](p232_007_fpga_bf16.tokens.npz)

The input is the repository's unmodified VoiceBank-DEMAND noisy test sample, with café noise at 12.5 dB nominal SNR; see its [source information](../../../../test_samples/p232_007.README.md) and [license](../../../../test_samples/p232_007.LICENSE.txt). Reconstruction files are derived from that sample.

## Accuracy interpretation

Waveform error is `||FPGA − CPU||₂ / ||CPU||₂`, using original sample indices with no gain, delay or polarity fitting. It is a numerical comparison, not a perceptual listening score. Different token choices and decoder arithmetic both affect it.

Decoding the **same FPGA-generated tokens** on CPU isolates decoder differences. Full-recording decoder error is **27.78% for IF8** and **34.95% for BF16**, each using its own generated IDs. Ordinary PyTorch BF16 decoding of the IF8-generated IDs differs from FP32 by **5.11%** ([CPU-only diagnostic](torch_bf16_diagnostic.json)). The additional FPGA error remains a limitation; these measurements do not establish a purely numerical or structural cause. End-to-end and same-token decoder errors use different reference waveforms, so their percentages do not add together.

On a separate quiet 0.4-second crop, IF8 matches **19/33** tokens with **31.54%** waveform error; BF16 matches **22/33** with **27.27%** error. Same-token decoder errors are **3.57% / 3.50%**. Processing RTF is **18.486 / 15.786**. This crop's FP32 output RMS is only 0.004277. Shorter crops are independent utterances with reset recurrent state and changed convolution context.

The quiet input is the first 400 ms of clean `p232_005.wav` from the [VoiceBank-DEMAND clean test archive](https://datashare.ed.ac.uk/server/api/core/bitstreams/dec213d3-bf57-4777-9663-c24bdce92d5e/content). The complete 48 kHz recording was resampled to 16 kHz with `librosa`/`soxr_hq`, then samples `[0:6400)` were saved as FLOAT WAV, without gain adjustment or added noise. Attribution: Cassia Valentini-Botinhao (2017), University of Edinburgh/CSTR, [DOI 10.7488/ds/2117](https://doi.org/10.7488/ds/2117), [CC BY 4.0](../../../../test_samples/p232_007.LICENSE.txt). The quiet reconstruction files derive from that crop. Its source and sample hashes are in [audio provenance](audio_provenance.json).

- [quiet_400ms_input.wav](quiet_400ms_input.wav)
- [quiet_400ms_cpu.wav](quiet_400ms_cpu.wav)
- [quiet_400ms_fpga_if8.wav](quiet_400ms_fpga_if8.wav)
- [quiet_400ms_fpga_bf16.wav](quiet_400ms_fpga_bf16.wav)
- [IF8 waveform comparison](p232_007_if8_comparison.json), [BF16 waveform comparison](p232_007_bf16_comparison.json), [IF8 decoder comparison](p232_007_if8_decoder_comparison.json), [BF16 decoder comparison](p232_007_bf16_decoder_comparison.json)

CPU decoder run records bind token hashes to reconstructed WAV hashes: [full IF8](p232_007_if8_tokens_cpu_decoder.metrics.json), [full BF16](p232_007_bf16_tokens_cpu_decoder.metrics.json), [quiet IF8](quiet_400ms_if8_tokens_cpu_decoder.metrics.json), [quiet BF16](quiet_400ms_bf16_tokens_cpu_decoder.metrics.json).

## Bin and execution

| Full-recording artifact | IF8 bytes | BF16 bytes |
| --- | ---: | ---: |
| Packed parameters and constants | 287,594,880 | 324,165,504 |
| Instruction program | 205,032,960 | 165,540,800 |
| Resident DRAM image | 492,627,840 | 489,706,304 |
| Serialized deployment bin | 492,668,127 | 489,746,351 |

IF8 contains **6,407,280 instructions** and BF16 **5,173,150**, each covering **180 graph operations**. Parameter storage includes packed weights, IF8 weight reuse, biases and constant tables. Generated binaries are excluded from Git; [IF8 compile metadata](full_compile_p232_007_final.json) and [BF16 compile metadata](full_compile_p232_007_bf16_tap.json) record their hashes.

Each FPGA benchmark invocation performs one model-image upload, one input upload, **one START**, **one HALT**, and one contiguous output read containing waveform and token data. Host neural operations: **0**. All recorded output samples and token lanes are finite; unused waveform lanes are zero. See the [IF8 run record](p232_007_fpga_if8.metrics.json), [BF16 run record](p232_007_fpga_bf16.metrics.json) and [CPU run record](p232_007_cpu.metrics.json).

All **80 software tests pass** ([test output](software_tests.txt)). The CPU reference matches the pinned upstream encoder, token IDs and forward waveform exactly on the [three recorded short fixtures](cpu_reference_validation.json). Hardware spot checks cover [317-step recurrent execution](native_long_lstm.json) and a [126,800-row dynamic clamp](native_large_clamp.json). A [resident-image reuse check](repeated_resident.json) runs speech, silence, then the same speech with one model upload and three START/HALT sequences. The repeated speech outputs and tokens are bit-identical, and silence produces finite output.

A [317-token hardware prefix check](quantized_features_317.json) compares the quantizer's output features with FP32 lookup of the exact selected IDs: 0.291% relative L2 error, with a maximum per-frame error of 0.407%. This supports correct ID-to-embedding selection; it does not resolve the decoder error above.

This model uses a 200-sample/12.5 ms token hop and centered convolutions. These measurements describe complete-utterance execution; they do not demonstrate independent 10 ms streaming.

Reproduction commands are in the [model README](../../README.md). [Artifact hashes](artifact_manifest.json) identify the collected files and implementation source. Raw run records retain the paths used during measurement; collected copies are identified by matching hashes.
