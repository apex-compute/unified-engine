# Initial BigCodec optimization estimates

**Superseded by the [validated hardware results](../20260914_optimized_italy/README.md): RTF 4.84 with bit-identical BF16 output, or 4.40 with IF8 recurrence.** The initial bins recorded below predate the hardware fixes for SRAM copy and final waveform layout. Recompile with the current source instead of deploying these initial candidates. This page preserves their original software measurements and traffic estimates.

Target: **RTF 4–5**, or **15.82–19.78 seconds** to process the same 3.955896-second recording. The [measured BF16 baseline](../20260914_italy/README.md) is **60.61 seconds / RTF 15.32**.

At this initial compilation stage, FPGA RTF and waveform accuracy were unmeasured. Italy reported image `0x364c8440` with invalid hardware-info value `0xdeadbeef`, and PCIe memory decoding subsequently became disabled. [Hardware readiness](hardware_readiness.json) preserves those readbacks. Hardware access was later restored on build `0x40519e0a`; the linked validated results cover the corrected implementation.

## Changes

- BF16 convolutions reuse overlapping SRAM input windows and resident weight strips. Dilated convolutions use decimated windows with the original tap order, removing intermediate im2col DRAM writes and reloads.
- Alias-free FIR filters and Snake arithmetic keep their intermediate values in SRAM. Native wide MAXPOOL comparisons replace identity-matrix clamps.
- LSTM gate and state updates remain in SRAM. Optional IF8 recurrent weights reduce the largest remaining weight stream; input projections and gate/state arithmetic stay BF16.
- Codebook selection keeps its first seven tournament rounds in SRAM. Comparisons retain the existing lowest-ID tie rule.
- Final mono Tanh processes one 32-byte AXI beat per padded sample row, reducing evaluated lanes from 64 to 16 and explicitly zeroing output padding.

The production runner retains **one resident program/parameter image, one input upload, one START/HALT sequence and one output-bundle read per utterance**. No host neural operations were added.

Modeled convolution DMA payload falls from **15.60 GB to 4.01 GB** (3.89× less), including weight reads of **7.84 GB to 1.18 GB**. The [traffic estimate](convolution_traffic_estimate.json) covers all 74 convolutions, including staging, bias and scatter, but excludes instruction fetch, compiler output zeroing, host transfers and other operator types. This is a traffic calculation, not a measured speedup.

## Compiled full-recording candidates

All candidates use BF16 convolutions and cover the same 180 operations, 63,400 padded samples and 317 tokens. MB below means 1,000,000 bytes.

| Recurrent weights | Parameters/constants | Program | Resident image | Instructions |
| --- | ---: | ---: | ---: | ---: |
| BF16 baseline | 324.166 MB | 165.541 MB | 489.706 MB | 5,173,150 |
| BF16 optimized | 331.252 MB | 83.658 MB | 414.909 MB | 2,614,302 |
| Encoder IF8, decoder BF16 | 312.967 MB | 81.953 MB | 394.921 MB | 2,561,046 |
| Encoder and decoder IF8 | 294.682 MB | 80.249 MB | 374.932 MB | 2,507,790 |

[Candidate records](candidate_artifacts.json) contain exact byte counts, hashes and passed artifact validation. Larger repeated activation constants improve tile reuse. Captured instruction counts do not directly predict execution time because device loops execute instructions repeatedly.

The two recurrent stacks read **23.93 GB** of BF16 recurrent weights per full recording. Encoder-only IF8 reduces this to **18.14 GB**; IF8 in both stacks reduces it to **12.34 GB**, including scales. These are byte-count calculations, not measured bandwidth or runtime.

## Validation and accuracy

All **93 software tests pass** ([test output](software_tests.txt)). Tests cover finite BF16 equivalence to the previous FIR/Snake and LSTM equations, dilation and transpose geometry, SRAM bank and tile bounds, codebook selection, IF8 packing against the driver, padding, and captured ISA jumps. Native MAXPOOL behavior, streamed IF8 arithmetic and complete-model speed still require hardware checks.

A [CPU weight-sensitivity experiment](if8_lstm_cpu_probe.json) changed only the four recurrent matrices, leaving the rest of the official model in FP32. Rounding these matrices to BF16 produced 0.065% / 2.277% waveform error on the quiet/full samples. IF8 quantization and dequantization produced 0.525% / 9.206% error; all 33/33 and 317/317 tokens matched FP32. These are CPU controls and do not establish FPGA accuracy. Input provenance is in the [baseline report](../20260914_italy/audio_provenance.json).

## Run after hardware is ready

From the repository root, compile and run the BF16 candidate:

```bash
python models/bigcodec/bigcodec_compile.py \
  --input test_samples/p232_007.wav --conv-precision bf16 \
  --lstm-precision bf16 --output /tmp/bigcodec-optimized.bin

python models/bigcodec/bigcodec_run_from_bin.py \
  --bin /tmp/bigcodec-optimized.bin \
  --input test_samples/p232_007.wav --output /tmp/bigcodec-optimized.wav

python models/bigcodec/bigcodec_compare.py \
  --reference models/bigcodec/validation/20260914_italy/p232_007_cpu.wav \
  --actual /tmp/bigcodec-optimized.wav \
  --reference-tokens models/bigcodec/validation/20260914_italy/p232_007_cpu.tokens.npz \
  --actual-tokens /tmp/bigcodec-optimized.tokens.npz \
  --report /tmp/bigcodec-optimized-comparison.json
```

Use `--lstm-precision encoder-if8` or `--lstm-precision if8` with a separate output bin to evaluate the recurrent-weight tradeoff. Generated bins are excluded from Git. The initial candidate paths above remain historical records; use the current compiler and the validated report's commands.
