# YOLOv5s

This directory contains YOLOv5s v7.0 object detection with precompiled
256x256, 320x320, 416x416, 512x512, 640x480, and 640x640 input profiles.
Convolution, SiLU, SPPF max-pooling, nearest-neighbor upsampling,
bottleneck residual additions, concatenation, and tensor handoff execute inside
one resident hardware program. Image preparation runs on the host before the
packed input upload; detection decoding, non-maximum suppression, and drawing
run on the host only after the bundled final output is read. The quantized CPU
fallback may still interpret the embedded graph operation by operation.

YOLOv5n has its own commands, configuration, cache, and documentation in
[`models/yolov5n`](../yolov5n). Its thin entrypoints reuse the graph,
quantization, artifact, and hardware primitives implemented in this directory.

## Hardware compatibility

YOLOv5 requires a convolution-enabled FPGA image. The repository-shipped
`update_19788da0.bin` image is **not compatible** because it does not implement
the required convolution/max-pool modes and registers.

The hardware runtime uses ordered queue-CONFIG geometry. The optimized
mixed-precision path additionally requires the corrected gather-IF8 scale
rewind introduced by Andromeda commit `77e8adf3`; older queue-CONFIG builds
`d93eea82`, `9ef15fc1`, and `663de8d5` do not provide that correction. The
four-channel banked gather and direct-bin path are strictly validated on
timing-clean RK-256 build `eed3a5d9` (WNS `+0.002 ns`, TNS `0`). The runtime
does not enforce an FPGA build-hash allow-list, so callers must select an image
with the required features. The validated build includes the read-only
`HW_INFO` register and remaps the live geometry CSRs. Queue-CONFIG direct
inference does not write those live CSRs. No compatible update image is shipped
in this repository.

Artifact-v6 schema/profile/digest checks, no-capture replay guards, offline
queue-CONFIG compilation, and quantized CPU direct execution are host-validated.
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
The cached checkpoint is stored as `yolov5s_bin/yolov5s-v7.0.pt` and is kept by
`make clean`.

The artifact compiler uses PyTorch's restricted `weights_only=True` path with a
minimal class allow-list, so it does not need an Ultralytics checkout, OpenCV,
ONNX, pandas, or seaborn. It downloads the upstream checkpoint only when the
local cache is missing; the direct artifact runner never downloads it. The
checkpoint is not part of this MIT-licensed
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

Run from the repository root. Compile the pinned checkpoint once, then execute
only from the generated artifact. The hardware runtime needs DMA device access;
the quantized CPU runtime does not:

```bash
# Downloads and verifies yolov5s-v7.0.pt if it is not already cached, folds
# Conv+BN, quantizes the 60 convolutions, and compiles each profile's combined
# fixed-address model/program deployment image offline into the artifact.
make yolov5s_bin

# Equivalent explicit compiler invocation; --output and --force are optional.
python3 models/yolov5s/yolov5s_compile.py \
  --output models/yolov5s/yolov5s_bin/yolov5s-andromeda.bin

# Hardware inference from that artifact. The default profile is 256x256.
python3 models/yolov5s/yolov5s_run_from_bin.py

# The same bin directly selects a precompiled rectangular or square profile.
python3 models/yolov5s/yolov5s_run_from_bin.py --resolution 640x480
python3 models/yolov5s/yolov5s_run_from_bin.py --resolution 640x640
python3 models/yolov5s/yolov5s_run_from_bin.py --list-resolutions

# Check the same artifact and graph on the quantized CPU backend, without FPGA.
python3 models/yolov5s/yolov5s_run_from_bin.py --cpu
```

`yolov5s-andromeda.bin` is the only model data file required by the direct
runner. Artifact format version 6 contains six validated fixed-shape graphs,
nibble-packed channel-IF4 and byte-packed gather-IF8 tensors for the CPU
fallback, and one fixed-address deployment image per profile containing both
packed parameters and its resident whole-graph program. During hardware-backend
initialization, the selected deployment image is uploaded exactly once. It is
never streamed instruction by instruction. The artifact also carries graph-validation metadata, section
digests, COCO names, anchors, strides, and preprocessing/postprocessing defaults.
Rebuild the artifact with `make yolov5s_bin FORCE=1` after this
resolution/schema change;
the regenerated image sizes and digests are pinned by the compiler rather than
carried over from the prior artifact.
The direct runner does not load or download the official `.pt` checkpoint,
quantize weights, repack static weights/scales/biases, or capture hardware
programs.

Each hardware inference has a strict four-step boundary: exactly one packed image
upload, one program kick, uninterrupted execution to the final HALT, and exactly
one bundled final-output read. The host performs no graph walking, operation
dispatch, tensor handoff, intermediate read, or intermediate write while the
program is running. Concatenation and all other graph-internal data movement stay
on the device. Convolution and max-pool geometry is carried in ordered CONFIG
opcode `0xC`, subtype `0`, so direct inference also performs no live
geometry-register writes. Only preprocessing before the input upload and decode,
NMS, and drawing after the final-output read remain host-side.

The annotated image is written under `yolov5s_bin/`. The script also emits one
machine-readable line for the automated test harness, for example:

```text
TEST_RESULT: {"model":"yolov5s","precompiled":true,"artifact_version":6,"full_graph":true,"model_upload_writes":1,"input_upload_writes":1,"program_kicks":1,"output_reads":1,...}
```

The canonical repository harness compiles or validates the artifact, then runs
the direct-bin entry point:

```bash
make model_test yolov5s
make model_test yolov5s run_from_bin  # reuse a valid cached artifact
```

The `run_from_bin` word only prevents the Makefile's pre-test clean. The
`yolov5s` entry always uses the direct runner, and its compile step reuses an
existing valid artifact. YOLOv5 remains opt-in because no compatible FPGA
update image is bundled, so it is not included when `model_auto_test.py` runs
the default suite.

Run the host-only quantization, planner, geometry, precompiled-image/digest,
no-capture replay, decode, and NMS regressions with:

```bash
python3 models/yolov5s/test_yolov5_helpers.py
```

`make clean` removes the generated single artifact, temporary program/parameter
artifacts, and annotated hardware outputs, but deliberately preserves the
checksum-verified official checkpoint. Use the `run_from_bin` Make modifier when
testing an already-built artifact so the pre-test clean does not remove it.

## Hardware execution accounting

The selected immutable deployment image is a backend initialization cost, not
per-inference traffic. A hardware inference consists
only of one packed input upload, one program kick, uninterrupted device
execution, and one bundled final-output read. Accelerator latency is measured
from that single kick to the final HALT. End-to-end timing may additionally
include the one input upload, the one final-output read, preprocessing, decode,
NMS, and drawing, but never host-side layer dispatch or intermediate tensor
traffic. Measurements from the retired multi-dispatch path are not comparable
and are intentionally not reported here.

## Layout

- `yolov5_common.py` - restricted checkpoint loader, graph interpreter,
  quantization, backends, preprocessing, decode, NMS, and drawing.
- `yolov5_artifact.py` - versioned single-artifact schema, validation, graph
  export, and checkpoint-free graph execution.
- `yolov5_precompiled.py` - offline fixed-address whole-graph deployment-image
  compiler and one-kick/no-capture hardware backend.
- `yolov5s_compile.py` - build-time compiler/exporter for
  `yolov5s-andromeda.bin`.
- `yolov5s_run_from_bin.py` - direct single-artifact hardware/quantized-CPU CLI.
- `yolov5s_config.json` - pinned source, model geometry, and thresholds.
- `yolov5s_bin/` - downloaded checkpoint and generated outputs/artifacts.
