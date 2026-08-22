# YOLOv5n

This directory owns the YOLOv5n v7.0 public entrypoints, configuration, model
cache, and generated artifacts. The graph interpreter, artifact schema, mixed
channel-IF4/gather-IF8 quantization, and Andromeda primitive implementations are
shared with
[`models/yolov5`](../yolov5) so YOLOv5n and YOLOv5s cannot drift at the engine
boundary.

The model embeds letterboxed 256x256, 320x320, 416x416, 512x512, 640x480,
and 640x640 RGB profiles and uses the 80 COCO classes.
Convolution, SiLU, SPPF pooling, upsampling, and residual additions run through
native engine primitives. Concatenation, detection decoding, NMS, and drawing
remain host-side.

## Hardware compatibility

Hardware inference requires native CONV2D/MAXPOOL, ordered queue-CONFIG
geometry, and the corrected gather-IF8 scale rewind introduced by Andromeda
commit `77e8adf3`. Older queue-CONFIG builds `d93eea82`, `9ef15fc1`, and
`663de8d5`, as well as the bundled `update_cf133b89.bin`, are rejected before
optimized model execution. The four-channel banked gather, direct-bin
inference, and complete Andromeda hardware suite are strictly validated on
timing-clean RK-256 build `83c27ced` (WNS `+0.002 ns`, TNS `0`). That build includes
the read-only `HW_INFO` register and remaps the live geometry CSRs. Queue-CONFIG
direct inference does not write those live CSRs.

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
# Checkpoint-backed hardware inference.
python3 models/yolov5n/yolov5n_test.py

# Float or hardware-compatible quantized CPU inference.
python3 models/yolov5n/yolov5n_test.py --backend cpu
python3 models/yolov5n/yolov5n_test.py --backend cpu-quantized

# Compile and run the checkpoint-free artifact.
make yolov5n_bin
python3 models/yolov5n/yolov5n_run_from_bin.py

# Select another exact precompiled profile from the same artifact.
python3 models/yolov5n/yolov5n_run_from_bin.py --resolution 640x480
python3 models/yolov5n/yolov5n_run_from_bin.py --resolution 640x640
python3 models/yolov5n/yolov5n_run_from_bin.py --list-resolutions

# Validate the same artifact without an FPGA.
python3 models/yolov5n/yolov5n_run_from_bin.py --cpu
```

The dedicated commands are pinned to YOLOv5n and intentionally do not expose a
`--variant` option. Use the sibling `models/yolov5` commands for YOLOv5s.

The canonical v5 artifact is
`yolov5n_bin/yolov5n-andromeda.bin`. It contains six fixed-shape graphs,
quantized CPU fallback tensors, one shared fixed-address parameter image, and a
resident program with ordered queue-CONFIG instructions for each profile. The
runtime uploads only the selected program. Rebuild it with `make yolov5n_bin FORCE=1`
after this resolution/schema change; the regenerated image sizes and digests
are pinned by the compiler rather than carried over from the prior artifact.
Runtime loads this single file without downloading a checkpoint, repacking
static parameters, or capturing programs. The current graph still requires
multiple hardware dispatches and host tensor handoff; it is one model artifact,
not one accelerator launch.

## 256 x 256 performance

On timing-clean RK-256 build `83c27ced`, a strict direct-bin run with no
unknown-hardware override reported an FPGA-only time of `30.366 ms`
(`10,119,312` cycles) and an execution-wall time of `85.746 ms`. The one-time
immutable model upload was `18,313,456` bytes in exactly two writes and took
`7.111 ms`. The run detected seven `person` instances, led by confidence
`0.685195`.

Every embedded profile completed strictly from the same bin and detected
`person`. FPGA-only times were `47.103`, `79.274`, `119.004`, `139.247`, and
`185.306 ms` for 320x320, 416x416, 512x512, 640x480, and 640x640 respectively.

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

Harness commands:

```bash
make model_test yolov5n
make model_test yolov5n_run_from_bin run_from_bin
python3 models/yolov5n/test_yolov5n.py
```

## Layout

- `yolov5n_test.py` - dedicated checkpoint-backed hardware/CPU entrypoint.
- `yolov5n_compile.py` - dedicated single-artifact compiler.
- `yolov5n_run_from_bin.py` - dedicated checkpoint-free runtime.
- `yolov5n_config.json` - pinned model, checkpoint, thresholds, and FPGA hashes.
- `yolov5n_bin/` - retained checkpoint and generated artifact/output cache.
