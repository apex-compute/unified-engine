# Convolution models

Run all commands from Andromeda's `pcie_utils/` directory:

```bash
cd pcie_utils
python3 -m pip install -r requirements.txt
```

The model compilers download and verify pinned checkpoints, then generate local
single-bin deployments. Generated bins, cached checkpoints, reports and output
media are not committed. Compile a bin before using its direct runner; add
`--force` to rebuild it after compiler or driver changes.

| Model | Task | Guide |
| --- | --- | --- |
| YOLOv5s / YOLOv5n | Object detection | [YOLOv5s](yolov5s/README.md), [YOLOv5n](yolov5n/README.md) |
| U-Net | Car foreground segmentation | [U-Net](unet/README.md) |
| DPDFNet2 | Speech enhancement at 16 kHz or native 8 kHz | [16 kHz](dpdfnet/README.md), [8 kHz](dpdfnet8khz/README.md) |
| BigCodec | Speech compression and reconstruction | [BigCodec](bigcodec/README.md) |

Hardware runs require a compatible CONV/MAXPOOL queue-CONFIG FPGA image.
The runners use the loaded image; they do not program the board. Audio model
bins target RK AXI-256. Each model guide describes its precision, input format,
CPU comparison and execution boundary. These ports originated in unified-engine
at `254f1dcead2c53714865be8a34cfea23db866520`; no separate checkout is needed.
Model configs and BigCodec’s upstream manifest pin their original sources and
checkpoints.

Sample inputs ship in `models/test_samples/`: `vette.jpg` and `people.jpg` for
the vision models and `apex.wav` (5.1 s of speech, 48 kHz stereo) for the audio
models. The audio runners use `apex.wav` when `--input` is omitted.

Compile and run YOLOv5s:

```bash
python3 models/yolov5s/yolov5s_compile.py
python3 models/yolov5s/yolov5s_run_from_bin.py \
  --device rk --image models/test_samples/vette.jpg --resolution 640x480
```

Run model host tests without FPGA access:

```bash
(
  for suite in yolov5s yolov5n unet dpdfnet dpdfnet8khz bigcodec; do
    python3 -m unittest discover -s "models/$suite" -p 'test_*.py' || exit 1
  done
)
```
