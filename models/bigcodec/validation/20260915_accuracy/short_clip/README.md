# BigCodec short-clip error decomposition

**Both candidates regress the full waveform error on this clip:** old optimized BF16 **19.29%**, candidate with both LSTM stacks changed **21.87%**, candidate with only the decoder LSTM changed **20.01%**. Lower token-only and conditional decoder errors do not make either candidate a full-waveform accuracy improvement here.

The decoder-only candidate was selected for the longer noisy-recording evaluation. That selection does not remove its short-clip regression shown here.

[Input](audio/input.wav): VoiceBank-DEMAND `p232_007`, café noise, nominal 12.5 dB SNR; mono 48 kHz, 189,883 samples / 3.955896 s. The 16 kHz model uses 63,295 samples, padded to 63,400 / 317 tokens. [Official FP32 CPU reconstruction](audio/official_cpu.wav), [official tokens](tokens/official_cpu.tokens.npz), and [audio attribution/license](ATTRIBUTION.md) are included. BigCodec reconstructs compressed audio; it is not a denoiser.

| Run | Full error | Token-only error | Conditional decoder error | Matching tokens | FPGA / same-token CPU WAV |
| --- | ---: | ---: | ---: | ---: | --- |
| Old optimized BF16 | 19.29% | 43.12% | 34.95% | [301/317](tokens/old_baseline.tokens.npz) | [FPGA](audio/old_baseline_fpga.wav) / [CPU](audio/old_baseline_same_tokens_cpu.wav) |
| Candidate: both LSTM stacks | 21.87% | 16.62% | 12.47% | [307/317](tokens/split_both.tokens.npz) | [FPGA](audio/split_both_fpga.wav) / [CPU](audio/split_both_same_tokens_cpu.wav) |
| Candidate: decoder LSTM only | 20.01% | 14.18% | 12.29% | [310/317](tokens/split_decoder.tokens.npz) | [FPGA](audio/split_decoder_fpga.wav) / [CPU](audio/split_decoder_same_tokens_cpu.wav) |

Let `y` be the official FP32 full-model reconstruction, `z` the official FP32 decoder output for the run's **actual FPGA tokens**, and `f` the actual FPGA waveform. Full error is `||f-y||₂ / ||y||₂`; token-only error is `||z-y||₂ / ||y||₂`; conditional decoder error is `||f-z||₂ / ||z||₂`. Token counts compare code IDs at identical frame indices. All samples are finite and compared over the complete original duration, without alignment, gain or polarity fitting.

`z` differs between rows. The conditional decoder comparison therefore does not hold decoder inputs fixed across variants. It includes device codebook-feature/projection rounding and decoder arithmetic downstream of the selected code IDs. These measurements describe agreement with the CPU codec, not perceptual quality or denoising performance.

With `t=z-y` and `d=f-z`, the exact identity is `||f-y||₂² = ||t||₂² + ||d||₂² + 2⟨t,d⟩`. Error percentages do not add; the conditional decoder percentage also uses a different denominator. The old run's component errors strongly cancel:

| Run | Token error energy | Decoder error energy | Cross-term `2⟨t,d⟩` | Full error energy | Error cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old optimized BF16 | 189.262736 | 125.515985 | **−276.915856** | 37.862865 | −0.898330 |
| Candidate: both LSTM stacks | 28.114754 | 15.741091 | +4.816242 | 48.672087 | +0.114471 |
| Candidate: decoder LSTM only | 20.467187 | 15.359237 | +4.916881 | 40.743306 | +0.138658 |

Energies sum squared normalized audio samples; `||y||₂² = 1017.8849314437322`. [results.json](results.json) contains unrounded values, definitions, original/copy hashes and bin settings. [manifest.json](manifest.json) hashes every packaged file except itself. Exact FPGA and same-token CPU run records are linked from the results. The archived [initial decomposition](records/actual_split_error_decomposition.json) retains its original local paths and initial score-model predictions; the table uses measured FPGA tokens only.

The supplemental [score-tree diagnostic](records/vq_tree_comparison.json) reproduces all 317 actual token IDs for both candidates and explains the initial decoder candidate's one-ID model discrepancy as a BF19 reduction tie. Its `baseline`/`accurate` labels refer to encoder inputs for the decoder-only/both-stack candidates. It is a CPU translation of locally available older RTL compared with saved hardware outputs; the intermediate tensors needed to rerun that separate diagnostic are excluded.

All three runs used Italy's **Kintex UltraScale+ KU5P, RK AXI256**, hardware `0x40519e0a`. Each uploaded one resident program/parameter image and one whole-utterance input, executed one START/HALT, then read the waveform/tokens from DRAM once; CPU neural operations were zero. Convolution, activation and both recurrent weight matrices are BF16 in every row. Both candidates add centered VQ scores and high/low codebook weights. They use compensated cell/tanh arithmetic and fused gate projections in the specified LSTM stack(s). The old run uses the original arithmetic and quantizer. FIR ordering and the final waveform tanh remain the original implementation.

| Run | Parameter bytes | Program bytes | Resident bytes | Instructions |
| --- | ---: | ---: | ---: | ---: |
| Old optimized BF16 | 331,251,584 | 83,998,976 | 415,250,560 | 2,624,968 |
| Candidate: both LSTM stacks | 331,267,968 | 144,615,488 | 475,883,456 | 4,519,234 |
| Candidate: decoder LSTM only | 331,267,968 | 114,322,432 | 445,590,400 | 3,572,576 |

The checkpoint SHA256 is `1fba3806e87cc01c1a65bea22fa1becefbbf46881e4219593c4d9f3cf56206b9`. Bin SHA256s, in table order: `884e3ba74075e5b5b344e6c834894d7a7c8a30fd437835cbd47a9fe41ba70706`, `3921b10dfd5a0d8575af0a9c6422fb20f7e6363770a8d816fc07ee1420962aab`, `200351f4fab341135fcd9675522ce68dae5d2b40190b7398329591b9a67076dc`. All four hashes were checked against the local files during packaging. Large checkpoint/bin files are excluded.

Recompute every metric and verify the WAV/token/run-record hashes from the repository root (Python with NumPy and soundfile; no model or hardware needed):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python models/bigcodec/validation/20260915_accuracy/short_clip/reproduce.py
```

The reproducer also works after copying this complete folder elsewhere. To regenerate one conditional decoder waveform with the repository's [CPU dependencies](../../../requirements.txt) and pinned checkpoint:

```bash
python models/bigcodec/bigcodec_run_cpu.py --mode decode \
  --tokens models/bigcodec/validation/20260915_accuracy/short_clip/tokens/split_decoder.tokens.npz \
  --output /tmp/bigcodec-short-decoder.wav --metrics /tmp/bigcodec-short-decoder.metrics.json --threads 1
```

The saved CPU records identify PyTorch `2.12.0.dev20260405+cu128`, one math thread and exact token/checkpoint hashes. Regenerated WAV container bytes may differ; compare decoded sample arrays. The 3.956 s result does not establish accuracy across longer recordings.
