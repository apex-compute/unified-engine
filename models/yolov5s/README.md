# YOLOv5s

Run commands from `pcie_utils/`. Install the [model dependencies](../README.md)
first. YOLOv5s detects the 80 COCO classes and embeds six RGB input profiles:
256×256, 320×320, 416×416, 512×512, 640×480 and 640×640.

## Compile and run

```bash
python3 models/yolov5s/yolov5s_compile.py
python3 models/yolov5s/yolov5s_run_from_bin.py \
  --device rk --dev xdma0 --image models/test_samples/vette.jpg \
  --resolution 640x480 --output /tmp/yolov5s.jpg

# Quantized CPU reference without FPGA access.
python3 models/yolov5s/yolov5s_run_from_bin.py \
  --cpu --image models/test_samples/vette.jpg --resolution 640x480 \
  --output /tmp/yolov5s_cpu.jpg
```

The compiler downloads and verifies the official v7.0 checkpoint using the
SHA256 in [yolov5s_config.json](yolov5s_config.json). Checkpoints and generated
artifacts live in the local `yolov5s_bin/` directory. Use `--output` to select
a compiler destination and `--force` to rebuild. The runner accepts `--bin` and
`--list-resolutions`; it loads only the artifact, without downloading weights
or compiling the graph.

The v7 bin contains all profiles, quantized CPU tensors, and one fixed-address
parameter/program image per profile. The selected image is uploaded once.
Convolutions store one weight pattern per output-channel chunk and reuse it
on-chip across each spatial tile. Rebuild older bins with `--force`.
Each inference then performs one input upload, one START/HALT sequence and one
bundled output read. Convolution, SiLU, SPPF, upsampling, residuals and
concatenation remain on the FPGA. Letterboxing, detection decoding, NMS and
image annotation run on the host.

## Precision and hardware

Conv+BN is folded before quantization. Channel-layout convolutions use IF4;
the small-channel stem uses gather-layout IF8. MixMSE selects integer or
floating codebooks per block, with signed BF16 scales. Default confidence and
IoU thresholds are 0.25 and 0.45.

Hardware requires native CONV2D/MAXPOOL, ordered queue-CONFIG geometry, corrected
gather-IF8 scale rewind, bank-Z convolution weight reuse and `HW_INFO`.
Program a compatible image separately;
`--device` does not select or flash a bitstream. Use a driver that matches the
installed firmware. Accelerator timing covers the single START-to-HALT
interval; host timing also includes boundary transfers and image processing.

## Default Winograd compilation and measured performance

Both YOLOv5s and YOLOv5n default to Winograd for eligible layers. This produces
a separate, pinned v8 artifact. Compilers, direct runners and CI wrappers
accept `--native` to select the native v7 baseline. The direct runner selects
the optimized artifact by default; `--bin` executes the exact supplied artifact.
`--optimize` and `--winograd` explicitly select the optimized default. The
model-auto-test registry now exercises this optimized path. The
[updated four-model report](../benchmarks/README.md) includes fresh measurements
for both YOLOv5 sizes and YOLO26n/s, plus the measured 64-channel Winograd
regression that motivates retaining the current layer selector.

```bash
python3 models/yolov5s/yolov5s_compile.py --winograd
python3 models/yolov5n/yolov5n_compile.py --winograd
python3 models/yolov5s/yolov5s_run_from_bin.py \
  --bin models/yolov5s/yolov5s_bin/yolov5s-andromeda_winograd.bin \
  --device rk --dev xdma0 --image models/test_samples/vette.jpg \
  --resolution 640x640 --output /tmp/yolov5s-winograd.jpg
```

The emitter adapts F(2×2,3×3) from the repository's
[`winograd` branch, commit `7c7ff328`](https://github.com/apex-compute/unified-engine/commit/7c7ff3289d9b2c4bf1dbca25233cad6af569ad57).
It replaces eligible 3×3, stride-1, dilation-1 convolutions with SRAM input/output
transforms and 16 BF16 matrix products per 2×2 output tile (instead of 36 direct
channel dot products). Existing IF4/IF8 weights are dequantized and transformed
offline, retaining the original logical quantized weights for CPU reference.
Bias, SiLU, concatenated channel layouts and odd feature-map edges are handled
on device. Inference still uses one input upload, one START/HALT and one output
read; there are no intermediate host transfers.

Measured on November 8, 2026: one FPGA core, 333.25 MHz, AXI-512, firmware
`f1e5582c`. Each number is the median of seven START-to-HALT measurements after
two warmups, using the same `people.jpg` input for both paths. Compilation,
model uploads, boundary transfers and image processing are excluded. The
[raw report](benchmarks/winograd.json) also records individual samples, host
execution times including boundary transfers, program hashes and output errors.

| Input | YOLOv5s native → Winograd (ms) | Latency reduction | YOLOv5n native → Winograd (ms) | Latency reduction |
|---|---:|---:|---:|---:|
| 256x256 | 84.089 → 82.059 | 2.41% | 41.071 → 41.071 | 0% (native fallback) |
| 320x320 | 131.289 → 127.419 | 2.95% | 64.059 → 64.027 | 0.05% |
| 416x416 | 221.750 → 215.193 | 2.96% | 108.078 → 108.079 | 0% (native fallback) |
| 512x512 | 334.600 → 322.660 | 3.57% | 162.839 → 162.403 | 0.27% |
| 640x480 | 392.433 → 378.678 | 3.51% | 191.164 → 190.816 | 0.18% |
| 640x640 | 522.479 → 502.872 | 3.75% | 254.377 → 253.539 | 0.33% |

YOLOv5s replaces seven of its 60 convolutions. YOLOv5n replaces two when they
are profitable; its 256×256 and 416×416 profiles keep the native implementation.
The 128-channel 8×8 and 13×13 maps were slower with Winograd in the initial
measurements, so the selector retains native convolution for small maps with
partial edge tiles and for maps smaller than 100 pixels. Wider, 256-channel
layers can amortize the transforms even on small maps. These thresholds reflect
this board; YOLOv5n's overall improvement is marginal.

All 24 model/resolution/image comparisons (`people.jpg` and `vette.jpg`) retained
the same detection counts and matched every baseline detection by class at
IoU ≥ 0.5. Where Winograd was active, raw-head SNR was 50.15–52.85 dB against the
native FPGA path; the worst matched-box IoU was 0.926, and the largest confidence
change was 0.0078. This checks numerical drift on two images, not COCO mAP.
The native baseline remains available with `--native`. Winograd has a memory cost:
YOLOv5s's 256×256 deployment image grows from 4.01 to 10.23 MiB, and active
profiles reserve 32 MiB of transform scratch plus the padded input map.

To reproduce the comparison with firmware compatible with the active driver:

```bash
python3 models/yolov5s/yolov5s_compile.py --native --output /tmp/yolov5s-native.bin
python3 models/yolov5n/yolov5n_compile.py --native --output /tmp/yolov5n-native.bin
python3 models/yolov5s/benchmark_winograd.py --artifact-dir /tmp \
  --output /tmp/yolo-winograd.json
```

The recorded run used the `user_dma_core.py` and `pcie_lock.py` files from
Andromeda commit `f1e5582c`, matching the installed firmware; this checkout's
driver pins `927fd8d9`. No firmware was changed. The exact runtime driver hash
is included in the report. The emitter produces the same pinned programs with
both drivers. Keep the normal board lock enabled when benchmarking.

## Tests and source

```bash
python3 -m unittest discover -s models/yolov5s -p 'test_*.py'

# Also rebuild both variants, native and Winograd, and verify all six profiles against their pinned
# deployment hashes, using locally cached official v7.0 checkpoints.
ANDROMEDA_YOLO_CHECKPOINT_DIR=/path/to/checkpoints \
  python3 -m unittest discover -s models/yolov5s -p 'test_yolov5_canonical_compile.py'
```

The native adapter pins Ultralytics source commit
`35b48237aef6d71ca9de2c5dea345d7536eb7fa7` separately from the v7.0 checkpoint.

The checkpoint and derived artifacts are subject to upstream YOLOv5 v7.0
GPL-3.0 terms. [YOLOv5n](../yolov5n/README.md) shares this directory's graph,
quantization and artifact implementations.
