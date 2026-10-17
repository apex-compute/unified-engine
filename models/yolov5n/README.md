# YOLOv5n

Run commands from `pcie_utils/` with the [model dependencies](../README.md).
YOLOv5n uses the [YOLOv5s implementation](../yolov5s/README.md) for graph
compilation, mixed IF4/IF8 quantization and single-bin hardware execution.
Its configuration and checkpoint are separate.

```bash
python3 models/yolov5n/yolov5n_compile.py
python3 models/yolov5n/yolov5n_run_from_bin.py \
  --device rk --dev xdma0 --image models/test_samples/people.jpg \
  --resolution 256x256 --output /tmp/yolov5n.jpg

# Quantized CPU reference without FPGA access.
python3 models/yolov5n/yolov5n_run_from_bin.py \
  --cpu --image models/test_samples/people.jpg --output /tmp/yolov5n_cpu.jpg

python3 -m unittest discover -s models/yolov5n -p 'test_*.py'
```

The compiler verifies the official v7.0 checkpoint against
[yolov5n_config.json](yolov5n_config.json). Add `--force` to rebuild or `--output`
to choose another artifact path. The default cache is `yolov5n_bin/`.
The direct runner accepts `--bin` and does not load the checkpoint.

The same artifact holds 256×256, 320×320, 416×416, 512×512, 640×480 and 640×640
profiles; use `--resolution` or `--list-resolutions`. It detects 80 COCO classes.
The selected parameter/program image is uploaded once. Each inference uses one
input upload, one START/HALT and one bundled output read, with no intermediate
host transfers. Image preparation, detection decoding, NMS and drawing run on
the host.

Channel convolutions use IF4. `model.0`, `model.1` and `model.2.m.0.cv2` use
gather-layout IF8. Hardware requires native CONV2D/MAXPOOL, ordered queue-CONFIG,
corrected gather-IF8 scale rewind and `HW_INFO`; program that image separately.

These entrypoints always select YOLOv5n. Source pinning, timing definitions and
the source compatibility audit are described in the YOLOv5s guide. The official
checkpoint and derived artifacts are subject to upstream YOLOv5 v7.0 GPL-3.0
terms.
