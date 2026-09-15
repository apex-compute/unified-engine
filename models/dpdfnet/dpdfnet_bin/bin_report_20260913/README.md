# DPDFNet2 instruction/data bins and accuracy assessment

Inspected on Italy, 2026-09-13. This report describes the production RK-256
artifact used for the noisy-speech evaluation. No production model or code
was changed, and no new hardware run was needed for this inspection.

## Numerical issue versus structural problem

The cause of the large `p257_018` onset mismatch remains unresolved. Existing
baseline, optimized and repeated FPGA outputs are bit-identical as decoded
sample values. The recent software optimizations therefore did not introduce
this mismatch. Outputs are finite and the best diagnostic alignment lag is zero.
This does not rule out a shared operator, indexing or recurrent-state bug.

CPU-only counterfactuals on the same complete clip give the following relative
L2 differences from the original FP32 CPU output. The original CPU rerun was
bit-identical to its saved waveform. These modified CPU runs are diagnostic,
not complete FPGA arithmetic emulations.

| Configuration | Full-clip error | Onset error (0.593–0.793 s) |
|---|---:|---:|
| Actual FPGA | 24.774% | 42.877% |
| CPU with LayerNorm epsilon removed | 0.003724% | 0.006294% |
| CPU with recurrent state rounded to BF16 between frames | 2.240% | 3.847% |
| CPU with FP32 initializers and recurrent state rounded to BF16 | 2.351% | 4.048% |

All eight ONNX LayerNorm nodes request epsilon approximately 1e-5. The compiler
and shared LayerNorm primitive omit it; this is a confirmed semantic mismatch,
but its CPU counterfactual effect is far too small to reproduce this outlier.
Likewise, rounding constants/state alone does not reproduce the onset failure.
The experiments do not emulate IF8 quantized convolution weights, hardware
accumulation, LALU approximations or every intermediate BF16 rounding point.
Consequently neither hardware numerical sensitivity nor a structural bug is
established as the root cause.

The next discriminating check is a graph-intermediate capture around the onset,
using the same input and recurrent state in both implementations. Independently
replay the first substantially divergent operator with those captured operands
to distinguish arithmetic differences from wrong indexing, operands or state.

## Extracted binary regions

The deployed single `.bin` is a serialized container holding metadata and one
resident device image. Its instruction and pre-program data regions were
extracted directly, without recompilation. Concatenating parameter/data first
and instructions second exactly reconstructs the uploaded image.

| Extracted file | Bytes | MiB | Accelerator-memory load address |
|---|---:|---:|---|
| [dpdfnet2.instructions.bin](dpdfnet2.instructions.bin) | 1,288,448 | 1.229 | `0x90fceb80` |
| [dpdfnet2.params-data.bin](dpdfnet2.params-data.bin) | 16,575,360 | 15.807 | `0x90000000` |

The instruction region contains **40,264 instructions of 32 bytes**:
40,246 UE operations, 15 configuration instructions, two NOPs and one terminal
HALT. The operation manifest covers 472 ONNX nodes plus one state commit.

The parameter/data region contains packed weights, scales and biases, helper
constants/selectors, zero data, initial recurrent state and alignment padding.
Its size is not a count or size of learned parameters alone. Runtime activation
scratch is a separate memory arena.

- Combined deployment file: **17,984,019 bytes**.
- Uploaded resident image: **17,863,808 bytes**.
- Artifact format: `andromeda.dpdfnet2.streaming-v2`; precision: `BF16/IF8-INT`.
- Original combined file SHA-256: `4f540ae650c27d822f4da24cfb0a0ac7f9cf33e53e84cbd9ce2fc7f1cdd5fc77`.

All deployment checksums, program bounds, operation ranges, queue configurations
and jump checks pass the existing artifact validator. The raw files are for
inspection; the production runner still loads `../dpdfnet2-andromeda.bin` once
and runs one START/HALT sequence per audio frame. Load addresses above are
accelerator memory addresses, not MMIO register bases.

Exact sizes, region hashes and instruction counts: [report.json](report.json).

CPU diagnostic evidence and reproducible script: [cpu_counterfactuals.json](cpu_counterfactuals.json), [cpu_counterfactuals.py](cpu_counterfactuals.py).

## Rerun the CPU diagnostic

Run from the repository root after installing the DPDFNet requirements. The
diagnostic reuses the pinned `models/dpdfnet/dpdfnet_bin/dpdfnet2.onnx` and saved
`noisy_eval_20260913/results/p257_018/{noisy,cpu,fpga}.wav` files. Those generated
inputs are not included with this report; see the [model preparation](../../README.md)
and [noisy evaluation commands](../../NOISY_TESTS.md) to prepare them. Exact
reproduction requires the archived files matching the hashes in the evidence
JSON. The script rejects changed inputs or an unequal original CPU rerun.
It reads saved FPGA output and performs CPU inference only.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python models/dpdfnet/dpdfnet_bin/bin_report_20260913/cpu_counterfactuals.py
```
