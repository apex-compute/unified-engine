# YOLOv5n

This directory owns the YOLOv5n v7.0 public entrypoints, configuration, model
cache, and generated artifacts. The graph interpreter, artifact schema, mixed
channel-IF4/gather-IF8 quantization, and Andromeda primitive implementations are
shared with
[`models/yolov5s`](../yolov5s) so YOLOv5n and YOLOv5s cannot drift at the engine
boundary.

The model embeds letterboxed 256x256, 320x320, 416x416, 512x512, 640x480,
and 640x640 RGB profiles and uses the 80 COCO classes. Convolution, SiLU, SPPF
pooling, upsampling, residual additions, concatenation, and tensor handoff run
inside one resident hardware program. Image preparation remains host-side before
the packed input upload; detection decoding, NMS, and drawing remain host-side
only after the bundled final output is read. The quantized CPU fallback may still
interpret the embedded graph operation by operation.

## Hardware compatibility

Hardware inference requires native CONV2D/MAXPOOL, ordered queue-CONFIG
geometry, and the FPGA build stamped from Andromeda commit `b34d1ac8`, reported
by the hardware-version register as `0xb34d1ac8`. The runner checks this value
before uploading the deployment image. This build includes per-output-channel
CONV bias reuse, the corrected gather-IF8 scale rewind, the read-only `HW_INFO`
register, and the live geometry CSR remap. Queue-CONFIG direct inference does
not write those live CSRs. The bundled `update_19788da0.bin` is not compatible.

## Checkpoint and quantization

The integration pins the official Ultralytics YOLOv5
[`v7.0` release](https://github.com/ultralytics/yolov5/releases/tag/v7.0),
source commit `915bbf294bb74c859f0b41f1c23bc395014ea679`, and
[`yolov5n.pt`](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)
with SHA-256
`4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3`.
The restricted checkpoint loader verifies the checksum and canonical topology
before consuming tensors. The checkpoint is cached at
`yolov5n_bin/yolov5n-v7.0.pt` and retained by `make clean`.

Conv+BN is folded before weights are quantized. Normal channel-layout
convolutions use signed-scale IF4 blocks; `model.0`, `model.1`, and
`model.2.m.0.cv2` use gather-layout IF8 over flattened spatial/channel blocks,
with MixMSE choosing INT8 for nearly all selected gather blocks.
The repository fixture is `test_samples/people.jpg`: at 256 x 256, the
quantized CPU path detects seven `person` instances at the normal `0.25`
confidence threshold, with top confidence `0.690374`.

The checkpoint is not part of this MIT-licensed repository. Review the
upstream YOLOv5 v7.0 GPL-3.0 terms before redistributing the checkpoint or a
derived artifact.

## Usage

Run from the repository root:

```bash
# Compile and run the checkpoint-free artifact. This is the only hardware path.
make yolov5n_bin
python3 models/yolov5n/yolov5n_run_from_bin.py

# Select another exact precompiled profile from the same artifact.
python3 models/yolov5n/yolov5n_run_from_bin.py --resolution 640x480
python3 models/yolov5n/yolov5n_run_from_bin.py --resolution 640x640
python3 models/yolov5n/yolov5n_run_from_bin.py --list-resolutions

# Validate the same artifact without an FPGA.
python3 models/yolov5n/yolov5n_run_from_bin.py --cpu

# Export the final 8192 decoded events from the circular trace BRAM.
python3 models/yolov5n/yolov5n_run_from_bin.py \
  --trace-tail perf_logs/yolov5n
```

The trace command writes a CSV and a native Perfetto `.pftrace` file below the
requested directory. Trace-register reads happen after the single terminal
HALT and add host-side debug overhead; `FPGA execution time` remains based on
the hardware latency counter.

The dedicated commands are pinned to YOLOv5n and intentionally do not expose a
`--variant` option. Use the sibling `models/yolov5s` commands for YOLOv5s.

The canonical v6 artifact is
`yolov5n_bin/yolov5n-andromeda.bin`. It contains six fixed-shape graphs,
quantized CPU fallback tensors and one fixed-address deployment image per
profile containing packed parameters plus a resident whole-graph program with
ordered queue-CONFIG instructions. The hardware backend uploads the selected
deployment image once during initialization, never instruction by instruction.
Rebuild it with `make yolov5n_bin FORCE=1`
after this resolution/schema change; the regenerated image sizes and digests
are pinned by the compiler rather than carried over from the prior artifact.
Runtime loads this single file without downloading a checkpoint, repacking
static parameters, or capturing programs.

Each hardware inference then performs exactly one packed image upload, one
program kick, uninterrupted execution to the final HALT, and exactly one bundled
final-output read. The host performs no graph walking, operation dispatch, tensor
handoff, intermediate read, or intermediate write while the program is running;
all concatenation and graph-internal data movement stay on the device.

## Hardware execution accounting

The selected immutable deployment image is a backend initialization cost, not
per-inference traffic. Accelerator latency is measured
from the one program kick to the final HALT. End-to-end timing may additionally
include the one packed input upload, the one bundled final-output read,
preprocessing, decode, NMS, and drawing, but never host-side layer dispatch or
intermediate tensor traffic. Measurements from the retired multi-dispatch path
are not comparable and are intentionally not reported here.

Harness commands:

```bash
make model_test yolov5n
make model_test yolov5n run_from_bin  # reuse a valid cached artifact
python3 models/yolov5n/test_yolov5n.py
```

## Layout

- `yolov5n_compile.py` - dedicated single-artifact compiler.
- `yolov5n_run_from_bin.py` - dedicated checkpoint-free runtime.
- `yolov5n_config.json` - pinned model, checkpoint, and thresholds.
- `yolov5n_bin/` - retained checkpoint and generated artifact/output cache.
