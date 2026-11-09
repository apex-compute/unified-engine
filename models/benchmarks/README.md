# YOLO optimization metrics

Measured November 9, 2026 (UTC) on one FPGA core at 333.25 MHz, AXI-512, firmware
`f1e5582c`, using its matching driver. Batch size is one. Each latency is the
median of seven runs after two warmups, using `people.jpg`. FPGA timing covers
START-to-HALT and excludes compilation, model upload, boundary transfers,
image preparation and postprocessing. FPS is 1000 / FPGA latency, not an
end-to-end application throughput measurement.

## 640×640 comparison

| Model | Native FPGA ms | Optimized FPGA ms | Latency reduction | FPGA FPS | Host execution ms | Uploaded model/program MiB |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5n | 254.380 | 253.540 | 0.33% | 3.94 | 289.028 | 2.67 |
| YOLOv5s | 522.474 | 502.869 | 3.75% | 1.99 | 537.836 | 10.78 |
| YOLO26n | 460.316 | 400.098 | 13.08% | 2.50 | 433.232 | 5.91 |
| YOLO26s | 1036.890 | 825.517 | 20.39% | 1.21 | 859.335 | 18.13 |

Host execution includes input packing/upload, FPGA execution and output
read/unpack. It excludes model loading, image preparation and postprocessing.
Uploaded MiB covers the selected profile, not the complete six-profile bin.

## All supported resolutions

Each cell is native → optimized FPGA milliseconds.

| Input | YOLOv5n | YOLOv5s | YOLO26n | YOLO26s |
|---|---:|---:|---:|---:|
| 256x256 | 41.071 → 41.072 | 84.090 → 82.059 | 73.491 → 63.838 | 165.701 → 132.494 |
| 320x320 | 64.057 → 64.028 | 131.284 → 127.416 | 114.780 → 99.723 | 258.699 → 206.297 |
| 416x416 | 108.083 → 108.076 | 221.751 → 215.192 | 194.064 → 168.652 | 437.980 → 350.591 |
| 512x512 | 162.836 → 162.404 | 334.599 → 322.660 | 293.192 → 254.765 | 661.516 → 526.587 |
| 640x480 | 191.166 → 190.816 | 392.432 → 378.675 | 344.511 → 299.514 | 776.346 → 618.740 |
| 640x640 | 254.380 → 253.540 | 522.474 → 502.869 | 460.316 → 400.098 | 1036.890 → 825.517 |

YOLOv5n uses the native fallback at 256×256 and 416×416; their tiny timing
differences are noise. Its 640×640 gain is only 0.33%.

## What is optimized

YOLOv5n/s use the Winograd F(2×2,3×3) implementation adapted from branch
`winograd`, commit `7c7ff3289d9b2c4bf1dbca25233cad6af569ad57`, with native
convolution for layers where transforms cost more than they save. At 640×640,
two layers use Winograd in YOLOv5n and seven in YOLOv5s. A measured expansion
to 64-channel layers regressed whole-graph latency from 253.539 to 258.873 ms
for n, and from 502.869 to 513.881 ms for s, so the 128-channel threshold stays.
See the [threshold experiment](../yolov5s/benchmarks/channel_threshold_sweep.json).

YOLO26n/s additionally replace channel-selection convolutions with resident
strided copies and split depthwise convolutions into 64-channel blocks. This
eliminates multiplication of off-block zeros without requantizing the IF8
weights. Attention, depthwise execution and channel copies stay on the FPGA.
Padding is initialized on every execution; no intermediate data crosses PCIe.

At 640×640, YOLO26n replaces 28 channel selections and seven depthwise layers.
The block-depthwise path beats Winograd on its eligible layers, so the final
optimized n graph uses no Winograd layers. YOLO26s replaces 42 selections and
eight depthwise layers, and retains Winograd on six regular convolutions.

| YOLO26 model | Native ms | Winograd only ms | All optimizations ms |
|---|---:|---:|---:|
| YOLO26n | 460.316 | 451.974 | 400.098 |
| YOLO26s | 1036.890 | 957.757 | 825.517 |

YOLOv5 uses the existing IF4/gather-IF8 weights. YOLO26 uses IF8 weights, BF16
activations and FP16 attention scratch. Winograd transforms and products use
BF16. These are comparisons within each port; the two model families use
different precision policies.

Winograd runs SRAM transforms and BF16 matrix-multiply instructions; its selected
layers do not issue native CONV2D instructions. Other convolutions, including
the optimized 64-channel depthwise blocks, still use native CONV2D instructions.
All of these operations run on the FPGA.

## Accuracy and validation

All four models were checked on `people.jpg` and `vette.jpg` at all six input
profiles: 48 native/optimized comparisons. Every optimized run retained the
native detection count, and every native detection matched an optimized one
by class at IoU ≥ 0.5. These two-image checks do not measure COCO mAP.

For YOLO26n, the minimum matched IoU across profiles is 0.9932, raw-head SNR
is at least 45.94 dB, and the largest matched confidence change is 0.0234. For
YOLO26s, these are 0.9828, 45.32 dB and 0.1099. The largest s confidence change
is the car at 320×320; its box IoU remains 0.9983. At 640×640, n returns 16
people and one car, and s returns 15 people and one car on the respective images.

Both YOLO26 floating-point adapters and NMS-free decoders match official
Ultralytics inference on the same two images at every resolution. All six
native, Winograd and optimized artifacts rebuild to their pinned digests; the
optimized CPU graph is numerically identical to its native CPU reference.
74 host tests passed across the YOLO suites, including channel-copy tests
with poisoned padding, partial 80-channel depthwise blocks and artifact
tampering. All four optimized CI wrappers passed CPU smoke tests; all four
optimized hardware paths were exercised by the benchmark. After making the
optimized path the default, all four compilers and direct CPU runners were
checked in default/native modes, all four default CI wrappers passed, and all
four direct runners passed another FPGA smoke test without strategy flags.

The following figures are **published upstream reference metrics**, not
accuracy measured for these quantized FPGA artifacts. They refer to the
NMS-free, one-to-one head used by these ports.

| Model | COCO mAP50–95 (NMS-free) | Fused parameters | FLOPs at 640 |
|---|---:|---:|---:|
| YOLO26n | 40.1 | 2.4 M | 5.5 B |
| YOLO26s | 47.8 | 9.5 M | 20.9 B |

Source: [Ultralytics YOLO26 metrics](https://docs.ultralytics.com/models/yolo26/#performance-metrics).
Full COCO validation of the IF8/BF16 FPGA artifacts has not been performed.

## Reproduce

Compile and run each optimized model (requires firmware matching the driver):

```bash
python3 models/yolov5n/yolov5n_test.py
python3 models/yolov5s/yolov5s_test.py
python3 models/yolo26n/yolo26n_test.py
python3 models/yolo26s/yolo26s_test.py
```

Wrappers default to 640×480; add `--resolution 640x640` for the first table.
Add `--cpu` for the quantized reference without hardware. All four compilers,
direct runners and CI wrappers default to the optimized path. `--optimize`
selects it explicitly; `--native` selects the original convolution baseline.
YOLO26 also retains `--winograd` for the Winograd-only comparison; optimized
builds use separate `_optimized.bin` files. YOLOv5 `--optimize` is an alias for
`--winograd` and writes `_winograd.bin`. An explicit runner `--bin` executes
that artifact exactly and cannot be combined with strategy flags. The
model-auto-test registry exercises the default paths for all four models.

Rebuild native baselines and repeat the measured comparisons:

```bash
mkdir -p /tmp/yolo-baselines
for model in yolov5n yolov5s; do
  python3 models/$model/${model}_compile.py --native --output /tmp/yolo-baselines/$model-native.bin
done
python3 models/yolov5s/benchmark_winograd.py --artifact-dir /tmp/yolo-baselines \
  --output /tmp/yolov5-metrics.json
for model in yolo26n yolo26s; do
  python3 models/$model/${model}_compile.py --native
  python3 models/$model/${model}_compile.py --winograd
  python3 models/$model/${model}_compile.py --optimize
  python3 models/$model/benchmark_${model}.py --output /tmp/$model-metrics.json
done
```

[Summary JSON](yolo_optimization.json) records the comparison table. Raw
samples, output errors, clock and driver identity are in the
[YOLOv5 report](../yolov5s/benchmarks/optimized.json),
[YOLO26n report](../yolo26n/benchmarks/optimized.json) and
[YOLO26s report](../yolo26s/benchmarks/optimized.json).
