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

The v6 bin contains all profiles, quantized CPU tensors, and one fixed-address
parameter/program image per profile. The selected image is uploaded once.
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
gather-IF8 scale rewind and `HW_INFO`. Program a compatible image separately;
`--device` does not select or flash a bitstream. The runtime does not enforce
a build-hash allow-list. Accelerator timing covers the single START-to-HALT
interval; host timing also includes boundary transfers and image processing.

## Tests and source

```bash
python3 -m unittest discover -s models/yolov5s -p 'test_*.py'
```

The native adapter pins Ultralytics source commit
`35b48237aef6d71ca9de2c5dea345d7536eb7fa7` separately from the v7.0 checkpoint.

The checkpoint and derived artifacts are subject to upstream YOLOv5 v7.0
GPL-3.0 terms. [YOLOv5n](../yolov5n/README.md) shares this directory's graph,
quantization and artifact implementations.
