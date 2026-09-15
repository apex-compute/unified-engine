# Paired sigmoid gates

Keeping decoder LSTM sigmoid gates as BF16 high/low pairs reduces the 3.956-second clip’s full waveform error from **20.01% to 16.23%**. Error against the CPU decoder using the identical FPGA tokens falls from **12.29% to 6.12%**. Token IDs are unchanged; their contribution remains **14.18%**. The full result still exceeds 10%. This page records the original unrolled diagnostic; the [later production validation](../final/README.md) uses counted device loops with identical arithmetic.

| Measurement | Existing decoder | Paired gates |
| --- | ---: | ---: |
| Full waveform L2 versus official FP32 codec | 20.0069% | 16.2277% |
| Decoder L2 versus FP32 decoding of the same tokens | 12.2874% | 6.1219% |
| Matching official tokens | 310/317 | 310/317 |
| Processing RTF | 4.9053 | 4.9240 |
| PESQ-WB versus CPU reconstruction | 4.5264 | 4.5274 |
| STOI versus CPU reconstruction | 0.997917 | 0.997952 |
| ESTOI versus CPU reconstruction | 0.995424 | 0.995516 |
| Multi-resolution magnitude error | 5.3838% | 5.2775% |

Relative L2 uses original sample indices without gain, delay or polarity fitting. PESQ/STOI retain their standard internal processing and measure agreement with codec reconstruction here; they do not establish denoising quality or replace the 10% waveform target. Processing RTF excludes artifact loading and model upload; RTF 1 is real time. [Exact comparisons, startup-inclusive times and error decomposition](results.json).

The full native run used build `0x90f1f464`, AXI256, one model upload, one input upload, one START, one HALT and one output read, with zero CPU neural operations. Its bin contains **331,267,968 parameter bytes** and **169,181,184 instruction bytes**, totaling **5,286,912 instructions**. [Bin hash and sizes](bin_manifest.json), [compile receipt](records/p232_007_paired_gates.paired_compile.json), [native run](records/p232_007_paired_gates.metrics.json), [FPGA WAV](audio/p232_007_paired_gates.wav), [input](../../20260915_accuracy/short_clip/audio/input.wav), [official CPU WAV](../../20260915_accuracy/short_clip/audio/official_cpu.wav).

The isolated decoder control improves LSTM error **0.9541% → 0.4682%** and subsequent FP32-tail waveform error **10.4768% → 2.7248%**, at **1.9498 → 2.1662 seconds**. It uses a separate frozen token sequence and is not a full-model result. The baseline matches its previous native output bit for bit. The primitive test matches all **6,144** predicted high/low components; recovered sigmoid maximum absolute error is **0.0003257** across logits ±12 and **0.00002981** inside ±8. Inputs outside ±8 are clamped. [Native evaluation](records/decoder_gate_pair_native_evaluation.json).

Applying the same candidate to the encoder was rejected: the CPU control improves local LSTM precision but reduces token agreement **309/317 → 307/317**. These encoder controls use FP32 dot accumulation, not native matrix reduction. [Encoder evidence](records/encoder_gate_pair_control.json).

Recheck the packaged evidence from the repository root:

```bash
python models/bigcodec/validation/20260915_comparison_criteria/paired_gates/reproduce.py
# Optional speech/spectral metrics; install models/bigcodec/requirements-quality.txt.
python models/bigcodec/validation/20260915_comparison_criteria/paired_gates/reproduce.py --quality
# Also regenerate the isolated LSTM reference using the official checkpoint.
python models/bigcodec/validation/20260915_comparison_criteria/paired_gates/reproduce.py --lstm-tail
```

These commands are offline and read-only unless `--output` is supplied. The default and `--quality` need no checkpoint or ignored diagnostic files. `--lstm-tail` needs the hash-pinned official checkpoint and reconstructs its reference from committed token IDs.

[Source snapshots](source_snapshots/) preserve the exact experimental scripts, including original paths and imports; they are not standalone native reproduction commands. [Source/dependency manifest](source_manifest.json) records the copied hashes, omitted historical dependencies and compile-source verification. Historical JSON receipts remain byte-identical. Generated binaries and heavyweight trace caches are omitted. The original unrolled paired-gate program exceeded the model arena for the longest bus case. [Counted device loops](../loop/README.md) resolve that program-size limit; see the [final noisy-file results](../final/README.md).
