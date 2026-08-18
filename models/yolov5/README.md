# YOLOv5s

This directory contains YOLOv5s v7.0 object detection at a 640 x 640 input
resolution. Convolution, SiLU, SPPF max-pooling, nearest-neighbor upsampling,
and bottleneck residual additions execute through native engine primitives.
Image preparation, concatenation, detection decoding, non-maximum suppression,
and result drawing run on the host because the current full-tensor primitives
return a host CHW tensor after each layer.

## Hardware compatibility

YOLOv5 requires a convolution-enabled FPGA image. The repository-shipped
`update_cf133b89.bin` image is **not compatible** because it does not implement
the required convolution/max-pool modes and registers.

Both hardware entry points use ordered queue-CONFIG geometry and currently
accept verified builds `d93eea82` and `9ef15fc1`, plus Python-test-only
descendant `663de8d5` whose FPGA RTL is identical to `9ef15fc1`. Older
native-CONV builds support only live geometry CSRs and are not accepted by
either YOLO runner. No compatible update image is shipped in this repository.
Do not start hardware inference until the running FPGA reports an appropriate
verified build. The stale `0x52a71442` comment in the imported driver predates
the convolution RTL and is not compatible.

The artifact-v3 compiler, schema/digest checks, no-capture replay guards,
checkpoint queue-mode selection, and quantized CPU direct run have been
validated on the host. The canonical `vette.jpg` CPU run detects `car` at
confidence `0.356716`. Direct artifact hardware replay was validated on RK
builds `d93eea82` and `9ef15fc1`; the strict poisoned-DRAM run detects `car` at
confidence `0.355042`. The checkpoint-backed queue path was also validated on
RK build `9ef15fc1` with the same detection and confidence.

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
64-value block layout. IF4 MixMSE chooses INT4 or FP4 independently per
`(output channel, kernel row, kernel column, 64-channel tile)` block; the sign
of its BF16 scale selects the codebook in hardware. A single scale for the
whole layer is not accurate enough for the pretrained detector.

The default confidence and IoU thresholds are `0.25` and `0.45`. The model uses
the 80 COCO classes and a letterboxed 640 x 640 RGB input.

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

# Hardware inference from that artifact. Artifact v3 requires queued-CONFIG
# build d93eea82 (or a later explicitly verified queue-compatible build).
python3 models/yolov5/yolov5_run_from_bin.py

# Check the same artifact and graph on the quantized CPU backend, without FPGA.
python3 models/yolov5/yolov5_run_from_bin.py --cpu
```

Artifact metadata contains the verified FPGA build allow-list. After that list
changes, rebuild a cached artifact with `make yolov5s_bin FORCE=1`.

`yolov5s-andromeda.bin` is the only model data file required by the direct
runner. Artifact format version 3 contains the validated fixed-640 graph,
nibble-packed IF4 tensors for the CPU fallback, a fixed-address prepacked static
parameter image, a resident precompiled instruction image with ordered CONFIG
geometry, per-operation dispatch metadata, section digests, COCO names, anchors,
strides,
hardware compatibility metadata, and preprocessing/postprocessing defaults.
For the canonical build, the static parameter image is `16,548,664` bytes and
the instruction image is `46,720` bytes containing 72 resident programs and 69
queue-CONFIG instructions.
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
TEST_RESULT: {"decoded_text":"car","n_detections":1,"precompiled":true,"artifact_version":3,"geometry_abi":"conv-config-inst-v1","detections":[...]}
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

The 6x6 stem has folded bias plus fused SiLU, so it cannot use the primitive's
bias-free gather mode. At 640 x 640 the channels-mode planner stages about
167 MiB across 17,120 small tiles for that layer. Host-side planning and the
validated RK replay confirm it fits the configured DRAM arenas. It is the main
expected throughput bottleneck until gather gains a bias/activation
epilogue or the driver keeps activations on device across layers.

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
