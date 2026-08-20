# YOLOv5n

This directory owns the YOLOv5n v7.0 public entrypoints, configuration, model
cache, and generated artifacts. The graph interpreter, IF4 quantization,
artifact schema, and Andromeda primitive implementations are shared with
[`models/yolov5`](../yolov5) so YOLOv5n and YOLOv5s cannot drift at the engine
boundary.

The model uses a letterboxed 640 x 640 RGB input and the 80 COCO classes.
Convolution, SiLU, SPPF pooling, upsampling, and residual additions run through
native engine primitives. Concatenation, detection decoding, NMS, and drawing
remain host-side.

## Hardware compatibility

Hardware inference requires native CONV2D/MAXPOOL and ordered queue-CONFIG
geometry. Verified builds are `d93eea82` and `9ef15fc1`; `663de8d5` has the
same FPGA RTL as `9ef15fc1`. Older native-convolution builds and the bundled
`update_cf133b89.bin` are rejected before model execution.

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

Conv+BN is folded before weights are quantized in the hardware's signed-scale
IF4 block layout. The repository fixture is `test_samples/people.jpg`: the
quantized CPU path detects four `person` instances at the normal `0.25`
confidence threshold, with top confidence `0.568826`.

Direct artifact replay is FPGA-validated on RK build `9ef15fc1`. The strict
2 GiB poisoned-DRAM harness run detects four `person` instances, led by
confidence `0.560543`.

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

# Validate the same artifact without an FPGA.
python3 models/yolov5n/yolov5n_run_from_bin.py --cpu
```

The dedicated commands are pinned to YOLOv5n and intentionally do not expose a
`--variant` option. Use the sibling `models/yolov5` commands for YOLOv5s.

The canonical v3 artifact is
`yolov5n_bin/yolov5n-andromeda.bin`. It contains the graph and quantized CPU
fallback tensors plus a `16,082,992`-byte fixed-address parameter image and a
`43,136`-byte instruction image with 72 resident programs and 63 ordered
queue-CONFIG instructions. Runtime loads this single file without downloading
a checkpoint, repacking static parameters, or capturing programs. The current
graph still requires multiple hardware dispatches and host tensor handoff; it
is one model artifact, not one accelerator launch.

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
