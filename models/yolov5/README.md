# YOLOv5s

This directory contains YOLOv5s v7.0 object detection with precompiled
256x256, 320x320, 416x416, 512x512, 640x480, and 640x640 input profiles.
Convolution, SiLU, SPPF max-pooling, nearest-neighbor upsampling,
and bottleneck residual additions execute through native engine primitives.
Image preparation, concatenation, detection decoding, non-maximum suppression,
and result drawing run on the host because the current full-tensor primitives
return a host CHW tensor after each layer.

YOLOv5n has its own commands, configuration, cache, and documentation in
[`models/yolov5n`](../yolov5n). Its thin entrypoints reuse the graph,
quantization, artifact, and hardware primitives implemented in this directory.

## Hardware compatibility

YOLOv5 requires a convolution-enabled FPGA image. The repository-shipped
`update_cf133b89.bin` image is **not compatible** because it does not implement
the required convolution/max-pool modes and registers.

Both hardware entry points use ordered queue-CONFIG geometry. The optimized
mixed-precision path additionally requires the corrected gather-IF8 scale
rewind introduced by Andromeda commit `77e8adf3`; older queue-CONFIG builds
`d93eea82`, `9ef15fc1`, and `663de8d5` are rejected for this artifact. The
four-channel banked gather and direct-bin path are strictly validated on
timing-clean RK-256 build `83c27ced` (WNS `+0.002 ns`, TNS `0`). That build includes
the read-only `HW_INFO` register and remaps the live geometry CSRs. Queue-CONFIG
direct inference does not write those live CSRs. No compatible update image is
shipped in this repository.

Artifact-v5 schema/profile/digest checks, no-capture replay guards, checkpoint
queue-mode selection, and quantized CPU direct execution are host-validated.
At 256 x 256, the canonical `vette.jpg` quantized CPU run detects `car` at
about `0.51` confidence (the final low bits can vary with the host BF16 kernel).

## Model and weights

The integration is pinned to the official Ultralytics YOLOv5
[`v7.0` release](https://github.com/ultralytics/yolov5/releases/tag/v7.0),
source commit `915bbf294bb74c859f0b41f1c23bc395014ea679`. On first use it downloads
the official
[`yolov5s.pt`](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)
checkpoint and verifies SHA-256
`8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee`.
The cached checkpoint is stored as `yolov5_bin/yolov5s-v7.0.pt` and is kept by
`make clean`.

The checkpoint-backed entry point and artifact compiler use PyTorch's restricted
`weights_only=True` path with a minimal class allow-list, so they do not need an
Ultralytics checkout, OpenCV, ONNX, pandas, or seaborn. They download the
upstream checkpoint only when the local cache is missing; the direct artifact
runner never downloads it. The checkpoint is not part of this MIT-licensed
repository; review the upstream YOLOv5 v7.0 GPL-3.0 terms before redistributing
it or a derived artifact.

Every Conv+BatchNorm pair is folded, then quantized in the hardware's native
64-value block layout. Normal channel-layout convolutions use IF4; MixMSE
chooses INT4 or FP4 independently per `(output channel, kernel row, kernel
column, 64-channel tile)` block. The small-channel `model.0` stem instead uses
gather-layout IF8 over flattened `[kernel row, kernel column, channel]` blocks;
MixMSE selects INT8 for nearly all of those blocks. For both widths, the sign
of each BF16 scale selects the integer or floating codebook in hardware. A
single scale for the whole layer is not accurate enough for the pretrained
detector.

The default confidence and IoU thresholds are `0.25` and `0.45`. The model uses
the 80 COCO classes and letterboxes into the selected embedded RGB profile.

## Usage

Run from the repository root. The default hardware backend needs DMA device
access; the two CPU backends do not:

```bash
# Detect objects in test_samples/vette.jpg using ordered queue CONFIG geometry.
python3 models/yolov5/yolov5_test.py

# Use another image or thresholds.
python3 models/yolov5/yolov5_test.py \
  --image path/to/image.jpg --conf-thres 0.25 --iou-thres 0.45

# Validate the graph, decode, and post-processing without an FPGA.
python3 models/yolov5/yolov5_test.py --backend cpu

# Simulate the same block-quantized weights and BF16 layer boundaries on CPU.
python3 models/yolov5/yolov5_test.py --backend cpu-quantized
```

### Single-artifact runtime

Export the pinned checkpoint once, then run without the checkpoint or network:

```bash
# Downloads and verifies yolov5s-v7.0.pt if it is not already cached, folds
# Conv+BN, quantizes the 60 convolutions, prepackages the fixed-address parameter
# image, compiles all hardware programs offline, and writes the default artifact.
make yolov5s_bin

# Equivalent explicit compiler invocation; --output and --force are optional.
python3 models/yolov5/yolov5_compile.py \
  --output models/yolov5/yolov5_bin/yolov5s-andromeda.bin

# Hardware inference from that artifact. The default profile is 256x256.
python3 models/yolov5/yolov5_run_from_bin.py

# The same bin directly selects a precompiled rectangular or square profile.
python3 models/yolov5/yolov5_run_from_bin.py --resolution 640x480
python3 models/yolov5/yolov5_run_from_bin.py --resolution 640x640
python3 models/yolov5/yolov5_run_from_bin.py --list-resolutions

# Check the same artifact and graph on the quantized CPU backend, without FPGA.
python3 models/yolov5/yolov5_run_from_bin.py --cpu
```

Artifact metadata contains the required FPGA build allow-list. After that list
changes, rebuild a cached artifact with `make yolov5s_bin FORCE=1`.

`yolov5s-andromeda.bin` is the only model data file required by the direct
runner. Artifact format version 5 contains six validated fixed-shape graphs,
nibble-packed channel-IF4 and byte-packed gather-IF8 tensors for the CPU
fallback, one shared fixed-address prepacked static parameter image, and one
resident instruction image with ordered CONFIG geometry per profile. Only the
selected program is uploaded. It also carries per-operation dispatch metadata,
section digests, COCO names, anchors, strides, hardware
compatibility metadata, and preprocessing/postprocessing defaults. Rebuild the
artifact with `make yolov5s_bin FORCE=1` after this resolution/schema change;
the regenerated image sizes and digests are pinned by the compiler rather than
carried over from the prior artifact.
The direct runner does not load or download the official `.pt` checkpoint,
quantize weights, repack static weights/scales/biases, or capture hardware
programs.

This is one **precompiled model artifact**, not one accelerator launch. The host
loads and validates the artifact once; the hardware backend uploads its immutable
parameter and program images once. It then walks the embedded graph, stages only
image-dependent operands, and starts the
resident program for each convolution, max-pool, upsample, or add operation.
Concatenation, tensor handoff, decode, NMS, and drawing remain host-side. Multiple
host dispatches remain necessary for graph and tensor handoff; convolution and
max-pool geometry is now carried in ordered CONFIG opcode `0xC`, subtype `0`, so
direct inference performs no live geometry-register writes.

The annotated image is written under `yolov5_bin/`. The script also emits one
machine-readable line for the automated test harness, for example:

```text
TEST_RESULT: {"model":"yolov5s","precompiled":true,"artifact_version":5,"input_resolution":"640x480","input_shape":[3,480,640],...}
```

To run either path through the repository harness:

```bash
make model_test yolov5s                         # checkpoint-backed path
make model_test yolov5s_run_from_bin run_from_bin  # prebuilt single artifact
```

The `run_from_bin` word only prevents the Makefile's pre-test clean; the explicit
`yolov5s_run_from_bin` model name selects the direct runner. Build the artifact
first with `make yolov5s_bin`. Both YOLOv5 entries are opt-in because no
compatible FPGA update image is bundled, so neither is included when
`model_auto_test.py` runs the default suite.

Run the host-only quantization, planner, capability, precompiled-image/digest,
no-capture replay, decode, and NMS regressions with:

```bash
python3 models/yolov5/test_yolov5_helpers.py
```

`make clean` removes the generated single artifact, temporary program/parameter
artifacts, and annotated hardware outputs, but deliberately preserves the
checksum-verified official checkpoint. Use the `run_from_bin` Make modifier when
testing an already-built artifact so the pre-test clean does not remove it.

## 256 x 256 performance

The gather primitive retains the normal folded-bias and fused-SiLU epilogue,
so the 6x6 RGB stem uses predominantly INT8 gather blocks instead of repeatedly
padding three channels into channel-mode IF4 blocks. Build `83c27ced` advances
four activation channels per walker cycle through a four-bank LUTRAM patch
store, while ordered exact-tail CONFIG groups remove duplicated edge outputs.
A strict direct-bin run, with no unknown-hardware override, reported an
FPGA-only time of `84.342 ms` (`28,106,912` cycles) and an execution-wall time
of `161.972 ms`. The one-time immutable model upload was `16,940,472` bytes in
exactly two writes and took `6.359 ms`. The run detected `car` at confidence
`0.510560`.

The same strict artifact and FPGA build detected `car` at every embedded
profile. FPGA-only times were `131.265`, `222.030`, `335.200`, `392.472`, and
`523.181 ms` for 320x320, 416x416, 512x512, 640x480, and 640x640 respectively.

FPGA-only time uses the `HW_INFO`-reported 333.25 MHz clock and is the corrected
sum of queue-start-to-HALT accelerator latency
counters. Execution-wall time additionally includes host tensor packing,
dynamic DMA, 72 resident-program dispatches, and host concatenations. The
two static uploads are the only model-state writes: runtime does not load a
checkpoint, quantize or repack static parameters, or capture programs. Hardware
initialization still performs its DRAM self-test, and graph replay still uses
image-dependent DMA. Artifact loading, preprocessing, decode, NMS, and drawing
are outside both execution timers. The model meets the strict sub-100-ms FPGA
execution target on one engine.

## Layout

- `yolov5_test.py` - accelerator/CPU entry point and CLI.
- `yolov5_common.py` - restricted checkpoint loader, graph interpreter,
  quantization, backends, preprocessing, decode, NMS, and drawing.
- `yolov5_artifact.py` - versioned single-artifact schema, validation, graph
  export, and checkpoint-free graph execution.
- `yolov5_precompiled.py` - offline fixed-address params/program image compiler
  and no-capture hardware replay backend.
- `yolov5_compile.py` - build-time compiler/exporter for
  `yolov5s-andromeda.bin`.
- `yolov5_run_from_bin.py` - direct single-artifact hardware/quantized-CPU CLI.
- `yolov5_config.json` - pinned source, model geometry, thresholds, and hardware
  compatibility metadata.
- `yolov5_bin/` - downloaded checkpoint and generated outputs/artifacts.
