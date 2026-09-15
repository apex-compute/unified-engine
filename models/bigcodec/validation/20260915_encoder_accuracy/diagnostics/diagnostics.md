The matrix FIR path reduces one measured encoder activation error and improves the short waveform, with a substantial runtime cost. These records distinguish FPGA execution, software geometry checks and comparisons against previously captured FPGA values. Relative L2 means `||actual − reference||₂ / ||reference||₂`; the tables express it as a percentage.

For isolated operation 087 (`encoder.block.7`), all variants received the same official FP32 teacher input rounded to BF16, shaped `[317, 1536]`. The reference is the official FP32 activation output. Each fresh FPGA case used build `0x90f1f464`, AXI 256, one START/HALT, unchanged input, intact guards and zero channel padding. The fresh serial baseline also matched the historical native output bit for bit.

| Isolated activation | Output L2 error | FPGA time |
| --- | ---: | ---: |
| Serial FIR, DC-adjusted BF16 taps | 0.688814% | 7.707 ms |
| Matrix FIR, same DC-adjusted taps | 0.463255% | 25.368 ms |
| Matrix FIR, original taps split into BF16 high/residual | 0.451664% | 36.217 ms |

These are activation errors, not waveform errors. [Native loop record](op087_native_loop.json) includes the input, reference, output, checkpoint, program and source hashes. The unchanged Snake calculation sits between the upsample and downsample FIRs. The same-tap DC control establishes a benefit from changing the accumulation path; the original-tap split adds a smaller improvement on this input.

Direct matrix output removed the final transpose: split time fell from 44.976 to 36.106 ms with identical output. Interior hardware loops reduced program duplication and preserved those output bits, taking 36.217 ms. See the [corrected original run](op087_native_corrected.json), [direct run](op087_native_direct.json), [direct parity](fir_direct_native_parity.json) and [loop parity](fir_loop_native_parity.json).

The [first harness run is rejected](REJECTED_op087_invalid_identity.json): it uploaded an FP32 identity matrix where transpose required 8,192 bytes of BF16. Both matrix candidates returned zeros, with L2 error 100%. Its historical `PASS` checked guards and must not be interpreted as numerical success. The deployed compiler already packed the identity correctly. Corrected runs checked its raw bytes, required the baseline to match the historical native tensor, and rejected zero output for the nonzero reference. The [identity regression test](../../../test_bigcodec_filter.py) checks actual packed bytes.

The [coefficient record](coefficient_representation.json) compares the twelve original FP32 taps with their packed representations. Absolute coefficient-vector L2 error is `5.22242e-4` for ordinary BF16 and `3.02931e-7` for high plus residual. The residual remains quantized; this is not exact FP32 storage or a corresponding waveform improvement factor.

Software geometry checks use independent FP64 grouped convolutions with BF16 stores and the native elementwise rounding model. They do not emulate the matrix accumulator: corrected DC and split FPGA outputs differ from that oracle by 0.225827% and 0.239269% L2. A separate [arithmetic validation](fir_native_model_validation.json) compared the inferred BF19/BF20 dot model against saved native post-Snake values: zero mismatches among 1,512 DC K192 outputs and 1,512 split K384 outputs, covering first/interior/last tiles and eight fixed channels. This validates those captured normal-value cases. The [model provenance](fir_native_model_provenance.json) explicitly limits the RTL inference; it is not a proof for every dot shape or IEEE edge case.

The [production parity check](production_program_parity.json) rehashed both saved artifacts. Prototype and production have identical 443,979,712-byte resident images, including the 110,352,448-byte program; only their container metadata differs. [Long-image parity](production_long_image_parity.json) also holds for the compiled bus and cafe_low_snr artifacts. The [artifact audit](production_artifact_review.json) checked 3,448,514 instructions, one terminal HALT, no SWI, contiguous operation ranges and memory bounds. The [graph review](production_integration_review.json) checked 60 configurations across 400, 63,400 and 379,000 samples, BF16/IF8 convolutions and filter scopes/stages. These are software checks, not additional native runs. [Filter tests](production_filter_tests.log), [core tests](production_core_tests.log) and [compiler tests](production_compile_tests.log) record their completed checks.

The separate 3.955896-second speech clip uses complete FPGA encoder, quantizer and decoder execution. All rows below use build `0x90f1f464`, legacy Snake, the same decoder math profile and the frozen official CPU WAV, with no gain, delay or polarity fitting.

| Encoder FIR candidate | Waveform L2 | Matching tokens | Processing RTF |
| --- | ---: | ---: | ---: |
| Serial baseline | 20.0069% | 310/317 | 4.8671 |
| Original split matrix, both stages | 16.1045% | 307/317 | 6.4581 |
| Original split matrix, up only | 26.0388% | 307/317 | 5.6556 |
| Original split matrix, down only | 28.4950% | 307/317 | 5.6732 |
| Sorted DC, both stages | 31.7715% | 303/317 | 4.8731 |

[Exact short summary and record hashes](short_candidate_summary.json), [current-board baseline comparison](short_baseline_current.comparison.json), [matrix waveform comparison](../short/matrix.comparison.json). RTF is recorded audio processing time divided by audio duration, excluding artifact loading and model upload; real time requires RTF ≤ 1. Better local arithmetic and token match counts do not imply uniformly better waveforms.

The 23.68225-second bus pilot improved only slightly: 20.5506% to 20.3603% waveform L2, while processing RTF rose from 4.8240 to 6.4091. The [baseline](../../20260915_accuracy/decoder/fpga_bf16/bus.comparison.json) ran on `0x40519e0a`; the [matrix prototype comparison](bus_matrix_prototype.comparison.json) and [runtime record](bus_matrix_prototype.metrics.json) use `0x90f1f464`. This long baseline was not replayed on the new build. Broader results belong in the parent report; these diagnostics do not select a deployment profile.

[Copied-record provenance](copied_records.json) preserves the historical paths and SHA256 hashes. Large tensor captures, bins and exploratory harnesses are deliberately absent, so these archived native results cannot be regenerated from this directory alone. The production implementation and portable geometry tests are [bigcodec_filter.py](../../../bigcodec_filter.py) and [test_bigcodec_filter.py](../../../test_bigcodec_filter.py). From the repository root, rerun the CPU tests with the project dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s models/bigcodec -p test_bigcodec_filter.py
```
