# Reproduce the noisy-audio validation

[run_validation.py](run_validation.py) and [build_report.py](build_report.py) are portable copies of the validation utilities. They find the repository by walking their parent directories and use the Python interpreter that starts them. Use the repository's BigCodec Python environment, with [dependencies](../../../requirements.txt) and the pinned checkpoint installed.

From the repository root, preview all eight cases with both BF16 and IF8 recurrent weights:

```bash
python models/bigcodec/validation/20260915_accuracy/decoder/run_validation.py
```

The default only verifies references and prints commands. It creates no files, compiles nothing and opens no device. The plan prints a fresh ignored `output_root`, an explicit execution command, and a report-preview command. Compensated cell/tanh arithmetic and fused gates default to the **decoder LSTM only**; both recurrent stacks use the selected weight precision. Centered VQ scores and the compensated codebook are enabled.

Generated bins/logs stay under `models/bigcodec/bigcodec_bin/accuracy_20260915/reproduce_runs/work/`. Generated WAVs, tokens, records and reports stay under `reproduce_runs/results/`. The published measured files are protected. Every new run records its own source/settings/reference freeze, including these copied utilities. Passing the same output root resumes only that unchanged reproduction; it does not reuse the original measured batch's freeze.

To compile all 16 jobs without hardware, choose a new results directory:

```bash
python models/bigcodec/validation/20260915_accuracy/decoder/run_validation.py \
  --output-root models/bigcodec/bigcodec_bin/accuracy_20260915/reproduce_runs/results/my-run-01 \
  --compile-only
```

Then explicitly request hardware execution on Italy, reusing that directory and its frozen compiled bins:

```bash
python models/bigcodec/validation/20260915_accuracy/decoder/run_validation.py \
  --output-root models/bigcodec/bigcodec_bin/accuracy_20260915/reproduce_runs/results/my-run-01 \
  --execute
```

`--execute` also compiles missing bins. It serializes native runs through the existing runner's shared `/tmp/pcie_ci_hw_<hostname>.lock` and checks build `0x40519e0a` inside that lock, before reset/upload. An explicit `--expected-version` overrides the build ID for a fresh run; the validation profile still requires AXI 256 and 333.25 MHz. The workflow retains the measured single-upload/START/HALT/read contract and hash, shape, precision, capacity and comparison checks. It expects Italy CPU cores 6, 8, 9 and 10; it reuses the frozen official FP32 CPU references.

Preview accumulated measurements, then write the report only when all 16 jobs are complete:

```bash
python models/bigcodec/validation/20260915_accuracy/decoder/build_report.py \
  --output-root models/bigcodec/bigcodec_bin/accuracy_20260915/reproduce_runs/results/my-run-01
python models/bigcodec/validation/20260915_accuracy/decoder/build_report.py \
  --output-root models/bigcodec/bigcodec_bin/accuracy_20260915/reproduce_runs/results/my-run-01 --write
```

The report utility defaults to read-only preview. With no output root, it reports that no reproduction exists; it never selects or rewrites the published results automatically. `--write` rejects missing/incomplete batches and writes only inside the chosen ignored reproduction directory. Waveform pooling uses summed error/reference energies; RTF pooling uses summed processing/audio durations.

The published measurements were produced by the original `run_accuracy_batch.py` and `build_accuracy_report.py` recorded in their freeze/evidence files. These copies deliberately create a fresh run with different utility source hashes. They do not resume or rewrite that measured batch; reporting it continues to use its original builder.

Offline checks for path protection, default previews, source freezes, hash bindings and failed execution guards are in [test_reproduction_utilities.py](test_reproduction_utilities.py). These checks do not execute hardware or compile a model.
