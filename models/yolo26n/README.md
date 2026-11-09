# YOLO26n

Andromeda port of Ultralytics' [YOLO26](https://docs.ultralytics.com/models/yolo26/)
nano detector, released in January 2026. The pinned official checkpoint detects
80 COCO classes. This port executes the one-to-one, NMS-free head, with direct
box-distance regression (no DFL), C3k2 blocks, SPPF and spatial PSA attention.
It supports 256×256, 320×320, 416×416, 512×512, 640×480 and 640×640 RGB inputs.
The shared implementation also supports [YOLO26s](../yolo26s/README.md).
See the [four-model metrics](../benchmarks/README.md) for the current comparison.

## Compile and run

Run from the repository root with the existing model dependencies installed.
Compilation and inference do not require an Ultralytics installation.
The compiler, direct runner and CI wrapper default to the optimized artifact.
Use `--native` for the original baseline or `--winograd` for Winograd alone.

```bash
python3 models/yolo26n/yolo26n_compile.py
python3 models/yolo26n/yolo26n_run_from_bin.py \
  --image models/test_samples/people.jpg --resolution 640x640

# Quantized CPU reference, without FPGA access.
python3 models/yolo26n/yolo26n_run_from_bin.py --cpu

# Compare hardware raw heads with the CPU reference; report repeated timings.
python3 models/yolo26n/yolo26n_run_from_bin.py --compare --warmup 2 --repeat 7

# Winograd-only comparison, saved separately.
python3 models/yolo26n/yolo26n_compile.py --winograd
python3 models/yolo26n/yolo26n_run_from_bin.py \
  --bin models/yolo26n/yolo26n_bin/yolo26n-andromeda_winograd.bin

# Faster channel copies and block-depthwise execution, with Winograd where useful.
python3 models/yolo26n/yolo26n_compile.py --optimize
python3 models/yolo26n/yolo26n_run_from_bin.py \
  --bin models/yolo26n/yolo26n_bin/yolo26n-andromeda_optimized.bin
```

The compiler verifies the official v8.4.0 checkpoint SHA-256 and loads it through
a restricted, weights-only class allow-list. [yolo26n_config.json](yolo26n_config.json)
pins the checkpoint, source revision and compiled native/Winograd/optimized bundles.
Artifacts contain all six profiles, quantized weights, class names and resident
program images. Both the compiler and loader check the canonical bundle digest,
graph dependencies, memory layout, instruction jumps and program hashes.
Use `--force` to rebuild. Generated checkpoints, bins and annotated images are
ignored by Git under `yolo26n_bin/`.

The runner only needs the bin and input image. It accepts `--bin`, `--resolution`,
`--list-resolutions`, `--image`, `--output`, `--conf-thres` (default 0.25),
`--max-det` (up to 300), `--dev` and `--device`. Outputs include an annotated JPEG
and a machine-readable `TEST_RESULT` with detections, timing and transfer counts.
`--compare` reports numerical differences; it does not enforce an accuracy gate.

## Execution and precision

The complete 187-operation graph, including all four attention heads, stays on
the FPGA. Each inference uses one packed input upload, one START/HALT and one
bundled result read. The model/program image is uploaded once. Image preparation,
anchor decoding, class sigmoid, top-k selection and rendering run on the host.
Postprocessing follows the upstream one-to-one head without NMS.

Convolutions use IF8 weights and BF16 activations. Attention uses 32-dimensional
Q/K padded to 64 lanes, 64-dimensional V and FP16 scratch, with the original
1/√32 scale. Spatial positions are padded to multiples of 64 and masked before
softmax, including the 13×13, 15×20 and 20×20 feature maps.
The Winograd path dequantizes the same IF8 weights offline and uses
BF16 transforms and matrix products for its selected convolutions.

The native baseline expands depthwise kernels to dense kernels and uses 1×1
convolutions for channel selection. The default optimized path replaces 28 selections with
resident strided copies and seven depthwise layers with 64-channel blocks,
preserving the original IF8 codes/scales while skipping off-block zeros.
The smaller, 16-channel selections retain the native fallback to satisfy DMA
alignment. Padded output lanes are initialized on every run.

At 640×640, this reduces FPGA execution from 460.316 to **400.098 ms (13.08%)**,
or **2.50 FPGA FPS**. Execution including boundary transfers takes 433.232 ms.
The selected model/program image shrinks from 7.22 to 5.91 MiB. All of n's
previous Winograd candidates are depthwise layers, and the block-depthwise
implementation is faster, so the final optimized n graph has zero Winograd
layers. YOLO26s retains Winograd on six regular convolutions. The native and
Winograd-only paths remain available for comparison.

Across all twelve image/profile comparisons, the optimized n path retained
every native detection by class at IoU ≥ 0.5. Minimum matched IoU was 0.9932,
minimum raw-head SNR 45.94 dB, and maximum confidence change 0.0234. See
[the updated measurements](benchmarks/optimized.json). This port remains slower
than YOLOv5n on this FPGA; upstream CPU/GPU speed claims do not describe this port.

Hardware requires the CONV/MAXPOOL queue-CONFIG image plus the driver's attention
operations. Use a driver matching the installed firmware; these scripts do not
flash the board. The measured board used firmware `f1e5582c` and its matching
driver from the Andromeda repository. The checked-in driver's firmware check
remains enabled. Only a single FPGA engine is used.

## Measured Winograd effect

Measured November 9, 2026 (UTC), on one FPGA core at 333.25 MHz, AXI-512. Values are
medians of seven START-to-HALT samples after two warmups, on `people.jpg`, with
the same IF8 checkpoint and driver for both paths. Timing excludes compilation,
model upload, input/output transfers and host postprocessing.

| Input | Native (ms) | Winograd (ms) | Latency reduction |
|---|---:|---:|---:|
| 256×256 | 73.493 | 72.761 | 1.00% |
| 320×320 | 114.781 | 113.187 | 1.39% |
| 416×416 | 194.059 | 191.925 | 1.10% |
| 512×512 | 293.190 | 288.181 | 1.71% |
| 640×480 | 344.516 | 338.933 | 1.62% |
| 640×640 | 460.320 | 451.976 | 1.81% |

At 640×640, execution including boundary transfers takes 494.974 → 485.892 ms.
Four convolutions use Winograd at that resolution; the 256×256 and 416×416
profiles use two. At 640×640 the uploaded image grows from 7.22 to 10.14 MiB,
and Winograd needs about 32 MiB of scratch plus its padded input map.
The complete bundles occupy 41.67 MiB (native) and 57.18 MiB (Winograd).
Winograd alone is available through `--winograd`; the faster default uses
block-depthwise execution for these layers.

[Raw timings and quality comparisons](benchmarks/winograd.json) include every
sample, firmware/driver identity, bundle hashes, transport counts and comparisons
against floating-point and quantized CPU references. Reproduce with:

```bash
python3 models/yolo26n/benchmark_yolo26n.py \
  --output /tmp/yolo26n-benchmark.json
```

Build all three strategies (default, `--native` and `--winograd`) before running this benchmark. It includes the
optimized path in addition to the native and Winograd-only comparison above.

## Validation and limits

The floating-point adapter and NMS-free decoder were compared with the official
Ultralytics model on both repository images at all six resolutions. All raw
heads passed `atol=0.002, rtol=0.0001`; decoded boxes/scores passed
`atol=0.02, rtol=0.0001`.

All twelve image/profile combinations ran on the FPGA with both compilation
strategies. At 640×640, the floating-point model, quantized CPU reference and both
FPGA variants each detected 16 people in `people.jpg` and one car in `vette.jpg`.
Every native detection at that resolution matched the floating-point output by
class at IoU ≥ 0.5; the minimum matched IoU was 0.962.

Quantization and Winograd can move scores across the 0.25 threshold. For example,
at 256×256 the floating-point model detects nine people and the native FPGA path
detects eight. At 416×416, native detects fifteen people and Winograd fourteen.
Across the twelve native/Winograd comparisons, raw-head SNR is 43.88–49.07 dB,
minimum matched-box IoU is 0.9935, and maximum matched confidence change is 0.03703.
These are checks on two images, not a COCO mAP evaluation.

Host tests cover grouped convolutions, channel selection, attention scaling and
padding, decoding, checkpoint verification, canonical recompilation and artifact
tampering. The existing YOLOv5 host suites also pass after the shared compiler
changes.

```bash
OMP_NUM_THREADS=4 python3 -m unittest discover -s models/yolo26n -p 'test_*.py'

# Also rebuild and validate both canonical bundles from cached weights.
ANDROMEDA_YOLO26_CHECKPOINT=models/yolo26n/yolo26n_bin/yolo26n.pt \
OMP_NUM_THREADS=4 python3 -m unittest discover -s models/yolo26n -p 'test_*.py'
```

For the optional upstream parity test, check out Ultralytics at
`6bc84f1eda65fc8b0aa2a2c71b2eb60908b00711`, install its reference dependencies in
an isolated environment, and set `ANDROMEDA_ULTRALYTICS_SRC` to that checkout in
addition to `ANDROMEDA_YOLO26_CHECKPOINT`. The test verifies the checkpoint hash
before using the upstream loader. The normal compiler/runner does not import
that package.

`yolo26n_test.py` compiles if needed and runs the `model_auto_test.py` smoke test.
It accepts `--cpu`, `--force`, `--native`, `--winograd` and `--optimize`; hardware tests use
the existing `UE_CI_CONV_IMAGE=1` harness gate. The registry exercises the optimized default.

Sources: [architecture](https://github.com/ultralytics/ultralytics/blob/6bc84f1eda65fc8b0aa2a2c71b2eb60908b00711/ultralytics/cfg/models/26/yolo26.yaml),
[official checkpoint release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Upstream source and checkpoint licensing remains applicable.
