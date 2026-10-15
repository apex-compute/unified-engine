# Carvana U-Net

Run commands from `pcie_utils/` with the [model dependencies](../README.md).
This port implements the two-class RGB `milesial/Pytorch-UNet` checkpoint with
`bilinear=False`. It produces a foreground mask.

```bash
python3 models/unet/unet_compile.py --download --resolution 256x256
python3 models/unet/unet_run_from_bin.py \
  --image models/test_samples/vette.jpg --backend hardware \
  --device rk --dev xdma0 --resolution 256x256 --compare \
  --output /tmp/unet_mask.png

# CPU reference using the artifact's quantized weights.
python3 models/unet/unet_run_from_bin.py \
  --image models/test_samples/vette.jpg --backend cpu-quantized \
  --resolution 256x256 --output /tmp/unet_cpu_mask.png

python3 -m unittest discover -s models/unet -p 'test_*.py'
```

The compiler verifies the official checkpoint digest in
[unet_config.json](unet_config.json). `--checkpoint` selects a local checkpoint;
`--output` chooses the artifact destination and `--force` rebuilds it. Checkpoints
and generated files stay in `unet_bin/`. The direct runner accepts `--bin`.

Each v2 artifact holds one fixed resolution whose dimensions are multiples of
16. Recompile for another resolution. The bin contains folded Conv+BN weights,
packed parameters, CPU reference tensors and a complete program. Weights use
symmetric INT8 codes with negative BF16 IF8 scales.

Hardware executes 35 convolutions and four pools with one input upload, one
START/HALT sequence and one output read. Skip tensors and intermediate features
remain in device DRAM. Transpose-convolution phases write into their destination
skip-concat map. Initialization uploads the model once; runtime does not
quantize weights, compile or dispatch individual layers.

Hardware requires a compatible CONV/MAXPOOL queue-CONFIG image. The runtime reads
AXI width and clock from `HW_INFO`; `--device rk` selects the XDMA transport.
`--device bittware_512` selects the BittWare transport. Device selection does not
program the FPGA.

The image is resized to the compiled resolution using bicubic interpolation;
logits are resized back before argmax. `--compare` reports logit RMSE and mask
agreement against the quantized CPU reference. It is not a dataset Dice/IoU
measurement or a pass/fail gate. `--backend cpu-fp32` uses folded FP32 weights;
the quantized CPU backend is not a bit-exact BF19/BF20 simulator.

`model_upload_s` is one-time initialization. `fpga_execution_s_sum` measures the
single device execution; `execution_elapsed_s` includes boundary transfers.
The compiler's `--weight-reuse-pixels 1..16` trades deployment size for latency.
Larger resolutions must fit the fixed DRAM arenas.

For a trace, add `--trace-tail /tmp/unet_trace` to the hardware command. The trace
contains only the final 8,192 decode events, with layer labels. Inferred windows
include dispatch/fetch overhead and do not directly measure engine utilization.
Tracing adds host overhead; omit it when measuring host latency.
