# YOLO26s

The small variant of Ultralytics YOLO26, with the 80-class, NMS-free one-to-one
head. It shares the verified graph adapter, compiler and runtime with
[YOLO26n](../yolo26n/README.md). The graph has 203 operations and eight attention
heads. Supported inputs are 256×256, 320×320, 416×416, 512×512, 640×480 and 640×640.

## Compile and run

Run from the repository root with the existing dependencies installed; neither
compilation nor direct inference needs an Ultralytics installation.

```bash
python3 models/yolo26s/yolo26s_compile.py
python3 models/yolo26s/yolo26s_run_from_bin.py \
  --image models/test_samples/people.jpg --resolution 640x640

# Compile if needed and run the optimized CPU reference.
python3 models/yolo26s/yolo26s_test.py --cpu

# Also produce baselines for a three-way comparison.
python3 models/yolo26s/yolo26s_compile.py --native
python3 models/yolo26s/yolo26s_compile.py --winograd
python3 models/yolo26s/benchmark_yolo26s.py --output /tmp/yolo26s-benchmark.json
```

The compiler, direct runner and CI wrapper default to direct channel copies,
block-depthwise lowering and Winograd for eligible regular convolutions.
`--optimize` selects this explicitly; `--native` and `--winograd` select the
comparison paths. The default writes a separate `_optimized.bin` v2
artifact; native and Winograd-only builds remain v1. At 640×640 the optimized
graph uses 42 channel copies, eight block-depthwise layers and six Winograd
layers. IF8 weights retain their original quantization. Activations are BF16;
attention scratch is FP16, and Winograd uses BF16 transforms/matrix products.

The complete graph stays resident on one FPGA engine: one input upload, one
START/HALT and one bundled output read per image. Preprocessing and NMS-free
top-k decoding run on the host. `--compare` reports raw-head differences from
the quantized CPU reference. `--warmup 2 --repeat 7` reports repeated timings.

The official v8.4.0 checkpoint has SHA-256
`646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`.
[yolo26s_config.json](yolo26s_config.json) pins it, the source revision and all
three compiled bundles. The loader verifies the restricted weights-only
checkpoint, canonical artifact hash and hardware layout. Local weights, bins
and output images under `yolo26s_bin/` are ignored by Git.

Use firmware matching the active driver. Measurements below used `f1e5582c`
and its matching Andromeda driver, with the firmware check intact. These
scripts do not flash the board. The model-auto-test registry selects the
optimized wrapper behind `UE_CI_CONV_IMAGE=1`.

## Measured performance

November 9, 2026 (UTC): one core, 333.25 MHz, AXI-512, batch one. Medians of seven
START-to-HALT samples after two warmups; excludes model load and host transfers.

| Input | Native ms | Optimized ms | Latency reduction |
|---|---:|---:|---:|
| 256×256 | 165.701 | 132.494 | 20.04% |
| 320×320 | 258.699 | 206.297 | 20.26% |
| 416×416 | 437.980 | 350.591 | 19.95% |
| 512×512 | 661.516 | 526.587 | 20.40% |
| 640×480 | 776.346 | 618.740 | 20.30% |
| 640×640 | 1036.890 | 825.517 | 20.39% |

At 640×640, Winograd alone takes 957.757 ms; the fully optimized path reaches
**1.21 FPGA FPS**, or 859.335 ms including input/output transfers. The uploaded
model/program image is 18.13 MiB versus 20.06 MiB native. The complete six-profile
optimized artifact occupies 104.56 MiB. The tensor/scratch arena is 177.16 MiB.

[Raw measurements](benchmarks/optimized.json) include individual timings,
hashes, memory use and CPU/FPGA output comparisons. The
[four-model report](../benchmarks/README.md) compares both YOLOv5 sizes and both
YOLO26 sizes under the same conditions.

## Accuracy and tests

The floating-point graph and decoder match official Ultralytics inference on
both sample images at all six resolutions. The optimized CPU graph matches
the native quantized CPU graph exactly. All three bundles rebuild to their
pinned hashes in host tests.

Across twelve native/optimized FPGA comparisons, every native detection matched
by class at IoU ≥ 0.5, with unchanged counts. Minimum matched IoU was 0.9828,
minimum raw-head SNR 45.32 dB and maximum confidence change 0.1099. The largest
confidence change was the car at 320×320, whose box IoU was 0.9983. At 640×640,
the native and optimized paths detected 15 people and one car on the respective
images. These checks do not establish COCO mAP for the quantized port.

Ultralytics publishes 47.8 COCO mAP50–95 for the NMS-free reference model,
9.5 M fused parameters and 20.9 B FLOPs at 640. These describe the upstream
model, not measured FPGA accuracy. See the
[official metrics](https://docs.ultralytics.com/models/yolo26/#performance-metrics).

The shared host suite covers both sizes. To include canonical compilation for
both cached checkpoints, run:

```bash
ANDROMEDA_YOLO26_CHECKPOINT=models/yolo26n/yolo26n_bin/yolo26n.pt \
ANDROMEDA_YOLO26S_CHECKPOINT=models/yolo26s/yolo26s_bin/yolo26s.pt \
OMP_NUM_THREADS=4 python3 -m unittest discover -s models/yolo26n -p 'test_*.py'
```

For official parity checks, also set `ANDROMEDA_ULTRALYTICS_SRC` to an isolated
Ultralytics checkout at `6bc84f1eda65fc8b0aa2a2c71b2eb60908b00711` with its reference
dependencies installed. The test verifies the pinned checkpoint hash before
using the official loader. Upstream source/checkpoint licensing still applies.
