# BigCodec encoder precision

The encoder FIR change reduces the separate 3.956-second clip's waveform error from **20.01% to 16.10%**, with processing RTF rising from **4.87 to 6.46**. A faster tap-order option reduces pooled error across eight 20–24-second noisy files from **23.94% to 22.64%**, at RTF **4.82**; two noisy cases and the short clip regress.

These are comparisons against the same official FP32 CPU codec reconstruction, at original sample indices. Relative L2 is `sqrt(sum((FPGA-CPU)^2) / sum(CPU^2))`. RTF is processing seconds divided by audio seconds; real time requires RTF ≤ 1. BigCodec is an audio codec and does not remove background noise.

| Configuration | Test set | Waveform error | Processing RTF | Result |
| --- | --- | ---: | ---: | --- |
| Previous BF16 profile | Eight noisy files, 173.01 s total | 23.94% | 4.8242 | Baseline |
| Sorted encoder FIR | Same eight noisy files | 22.64% | 4.8250 | Six improve, two regress |
| Previous BF16 profile, replayed on current build | Separate 3.956 s clip | 20.01% | 4.8671 | Baseline |
| Matrix encoder FIR, original split coefficients | Same short clip | 16.10% | 6.4581 | Lower error; slower |

[Eight-file comparison and WAVs](sorted_comparison.md), [short-clip comparison](short_comparison.md), and [native layer diagnostics](diagnostics/diagnostics.md) include exact metrics and regressions. Matrix FIR has only the short clip and bus pilot measured so far; its full noisy-file validation is pending PCIe rescan after another CI job changed the board to AXI512. The correct AXI256 image has been restored over JTAG.

The new matrix kernel retains original filter coefficients as BF16 high values plus BF16 residuals. One native dot product accumulates all tap products before storing BF16. The tensor layout, padding, filter geometry and legacy Snake function remain unchanged. Parameters, activations and outputs remain BF16 representations; this does not provide FP32 arithmetic. Native matrix reduction uses BF19/BF20 internally. Small remaining encoder differences can change discrete codebook choices and amplify waveform error.

Compile the matrix filter profile for a file:

```bash
python models/bigcodec/bigcodec_compile.py \
  --input input.wav --conv-precision bf16 \
  --lstm-cell-precision compensated --lstm-tanh-precision compensated \
  --lstm-fused-gates --lstm-math-scope decoder \
  --center-quantizer-scores --compensated-codebook \
  --filter-accumulation matrix --filter-math-scope encoder \
  --memory-layout large-program \
  --output models/bigcodec/bigcodec_bin/input-matrix.bin

python models/bigcodec/bigcodec_run_from_bin.py \
  --bin models/bigcodec/bigcodec_bin/input-matrix.bin \
  --input input.wav --output reconstructed_fpga.wav \
  --expected-version 0x90f1f464
```

Use `--filter-accumulation sorted` for the faster optional ordering change. The compiler and existing deployment bin retain their previous defaults. Applying matrix precision only to the upsampling or downsampling stage regressed the short clip and is not recommended.

Every measured FPGA run uses one bin containing parameters and instructions, one model upload, one input upload, one START, one HALT and one output read, with zero CPU neural operations. This is a complete-utterance execution; independent 10 ms calls are not equivalent. BigCodec produces one token per 200 native samples, or 12.5 ms at 16 kHz. Processing RTF excludes loading/validating the bin and uploading its model image; startup-inclusive times are reported separately.

The short matrix image contains **333,627,264 parameter bytes** and **110,352,448 instruction bytes**, with **3,448,514 instructions**. Its production and tested prototype images are byte-identical. Long-file sizes and hashes are recorded in the [compile catalog](matrix_compile_catalog.json); generated bins stay in the ignored model cache.

All **161 software tests pass**; [test command and log](tests.json). [Hardware restoration receipt](hardware_restore.json) records the verified artifact and JTAG operation.
