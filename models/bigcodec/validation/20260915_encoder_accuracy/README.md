# BigCodec encoder precision

Sorted encoder FIR accumulation gives the best pooled error across the eight 20–24-second noisy files: **22.64%**, down from **23.94%**, at processing RTF **4.825**. Two noisy files and the separate short clip regress. The slower matrix filter reaches **23.85%** at RTF **6.409** and is not a better overall choice. Neither option replaces the existing deployment defaults.

These are comparisons against the same official FP32 CPU codec reconstruction, at original sample indices. Relative L2 is `sqrt(sum((FPGA-CPU)^2) / sum(CPU^2))`. RTF is processing seconds divided by audio seconds; real time requires RTF ≤ 1. BigCodec is an audio codec and does not remove background noise.

| Configuration | Test set | Waveform error | Processing RTF | Result |
| --- | --- | ---: | ---: | --- |
| Previous BF16 profile | Eight noisy files, 173.01 s total | 23.94% | 4.8242 | Baseline |
| Sorted encoder FIR | Same eight noisy files | 22.64% | 4.8250 | Six improve, two regress |
| Matrix encoder FIR, original split coefficients | Same eight noisy files | 23.85% | 6.4088 | Six improve, two regress |
| Previous BF16 profile, replayed after board restoration | Separate 3.956 s clip | 20.01% | 4.9053 | Baseline |
| Sorted encoder FIR | Same short clip | 31.77% | 4.8731 | Regressed |
| Matrix encoder FIR, original split coefficients | Same short clip | 16.10% | 6.4573 | Lower error; slower |

[Eight-file comparison and WAVs](matrix_comparison.md), [short-clip comparison](short_comparison.md), and [native layer diagnostics](diagnostics/diagnostics.md) include exact metrics. Sorted accumulation regresses bus and low-SNR public square; matrix accumulation regresses office and public square. Public-square error rises from **29.77% to 37.05%** with matrix filters, despite more matching tokens. The short-clip improvement alone is not sufficient evidence for selecting that profile.

The restored AXI256 board reproduced the serial baseline and production matrix short outputs bit for bit. All eight matrix bins completed on build `0x90f1f464` at 333.25 MHz. The cheaper original-BF16-only matrix candidate regressed the short clip to 33.78% at RTF 5.83 and was rejected; see the [restored-board short comparison](restored_short_comparison.md).

[Short-only speed trials](fir_speed_comparison.md) also tested smaller matrix tiles and fused transpose writeback. Smaller tiles reached 19.14% error at RTF 6.584, worse and slower than the matrix control. Fused writeback preserved its waveform and tokens exactly and reduced RTF from 6.457 to 6.360, about 1.5%. These prototypes were not promoted: they do not establish a better eight-file accuracy/runtime result, and their code remains outside the production compiler.

The new matrix kernel retains original filter coefficients as BF16 high values plus BF16 residuals. One native dot product accumulates all tap products before storing BF16. The tensor layout, padding, filter geometry and legacy Snake function remain unchanged. Parameters, activations and outputs remain BF16 representations; this does not provide FP32 arithmetic. Native matrix reduction uses BF19/BF20 internally. Small remaining encoder differences can change discrete codebook choices and amplify waveform error.

Compile the faster optional sorted filter profile for a file:

```bash
python models/bigcodec/bigcodec_compile.py \
  --input input.wav --conv-precision bf16 \
  --lstm-cell-precision compensated --lstm-tanh-precision compensated \
  --lstm-fused-gates --lstm-math-scope decoder \
  --center-quantizer-scores --compensated-codebook \
  --filter-accumulation sorted --filter-math-scope encoder \
  --memory-layout large-program \
  --output models/bigcodec/bigcodec_bin/input-sorted.bin

python models/bigcodec/bigcodec_run_from_bin.py \
  --bin models/bigcodec/bigcodec_bin/input-sorted.bin \
  --input input.wav --output reconstructed_fpga.wav \
  --expected-version 0x90f1f464
```

Use `--filter-accumulation matrix` for the matrix experiment. The compiler and existing deployment bin retain their previous defaults. Applying matrix precision only to the upsampling or downsampling stage also regressed the short clip.

Every measured FPGA run uses one bin containing parameters and instructions, one model upload, one input upload, one START, one HALT and one output read, with zero CPU neural operations. This is a complete-utterance execution; independent 10 ms calls are not equivalent. BigCodec produces one token per 200 native samples, or 12.5 ms at 16 kHz. Processing RTF excludes loading/validating the bin and uploading its model image; startup-inclusive times are reported separately.

The short matrix image contains **333,627,264 parameter bytes** and **110,352,448 instruction bytes**, with **3,448,514 instructions**. Its production and tested prototype images are byte-identical. Long matrix bins share the same parameter section and use **553,391,040–637,939,776 instruction bytes**. Sizes and hashes are recorded in the [compile catalog](matrix_compile_catalog.json); all eight bins and their parameter/program sections were rehashed during report generation. Generated bins stay in the ignored model cache; the [offline comparison manifest](matrix_manifest.json) uses tracked WAVs, tokens and metrics only.

All **161 software tests pass**; [test command and log](tests.json). [Hardware restoration receipt](hardware_restore.json) records the verified artifact and JTAG operation.
