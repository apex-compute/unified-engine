# Carvana U-Net: single-bin Andromeda deployment

Implements the `milesial/Pytorch-UNet` two-class, RGB, `bilinear=False`
checkpoint topology. Hardware executes 35 convolutions and 4 pools in one
precompiled program, with one kick and one terminal HALT. Intermediate
features and skips remain in device DRAM. Four transpose-convolution phases
scatter directly into the skip-concat tensor. Only image preparation and
final mask postprocessing remain host-side; there are no intermediate host
transfers. CPU comparison is a separate run outside execution timing.

BN is folded in inference mode with epsilon 1e-5. Weights use symmetric INT8
codes (IF8 decoder with negative BF16 scales), one magnitude per output
channel. The v2 artifact contains packed parameters, a precompiled program,
tensor/operation manifests and CPU reference tensors. Runtime does not
quantize, compile or dispatch layers. Hashes and memory layouts are validated
before upload. Only trusted artifacts/checkpoints should be loaded.
CPU quantized inference is not bit-exact BF19/BF20 simulation.

```bash
# From the repository root, with its Python environment activated:
python models/unet/unet_compile.py --download --force --resolution 256x256
python models/unet/unet_run_from_bin.py --image test_samples/vette.jpg \
  --backend cpu-quantized --resolution 256x256 --progress
python models/unet/unet_run_from_bin.py --image test_samples/vette.jpg \
  --backend hardware --device bittware_512 --dev xdma0 \
  --resolution 256x256 --compare --progress
python -m unittest discover -s models/unet -p 'test_*.py'
```

Use `--checkpoint PATH` for an existing official checkpoint, `--force` to
rebuild. The default official checkpoint SHA256 is pinned in the config;
custom checkpoint digests are recorded in the artifact.
Never pass untrusted pickle checkpoints; loading uses `weights_only=True`.

The new format is `andromeda.unet.whole-graph-v2`. Old layer-v1 bins must be
rebuilt for hardware execution (CPU compatibility is retained). Each bin holds
one fixed resolution; both dimensions must be multiples of 16. Compile with
`--resolution WIDTHxHEIGHT` and optionally `--output` for another profile.
The runner rejects a hardware/profile mismatch before opening the device.
Larger shapes must fit fixed DRAM arenas. Physical validation is on BITTWARE
AXI512 at 256x256; AXI256-compatible alignment is used but not yet tested on
an AXI256 board for this model.

The allocator reuses tensor storage only after its final consumer, including
skips. Transpose phases write directly into the final skip-concat map, avoiding
four temporary phase tensors and their copy pass. Repeated weight streams have
a 4 MiB floor and default to a twelve-spatial-copy budget for deep kernels;
`--weight-reuse-pixels 1..16` exposes the deployment-size/latency tradeoff.
The 256x256 default uses about 420 MiB of the 512 MiB model arena. OC32 output is staged in
device scratch and written with individual contiguous 64-byte transfers;
multi-pixel 64-byte strided writes were incorrect on the local AXI512 build.
Instruction layout follows the existing YOLO static tile emitter; hardware
loops are not disabled globally. This is not a globally optimal scheduler.
Device staging can increase FPGA cycles despite reducing host latency.

`model_upload_s` measures one-time backend initialization, excluded from
`execution_elapsed_s`. Each execution performs one input upload, one program
kick and one output read. `fpga_execution_s_sum` is retained for compatibility
but now measures a single hardware execution.

The explicit resolution resizes the image with bicubic interpolation. It
does not reproduce upstream's image-relative `scale=0.5` unless sizes agree.
Logits are resized back to original dimensions before argmax; output is a
black/white foreground mask. `--compare` reports logit RMSE and mask agreement
against the quantized reference, not dataset Dice/IoU or a pass/fail verdict.
Use `--backend cpu-fp32` for the folded full-precision reference.

## Hardware trace

```bash
python models/unet/unet_run_from_bin.py \
  --image test_samples/vette.jpg --backend hardware \
  --device bittware_512 --dev xdma0 --resolution 256x256 --compare \
  --trace-tail perf_logs/unet_single_bin
```

The file is `perf_logs/unet_single_bin/unet_256x256_tail_perfetto.pftrace`
with a neighboring CSV. This is the native hardware timeline after one HALT,
not concatenated per-layer runs.
Events are prefixed with layer names, including pooling and transpose phases.
The DECODE_GAPS track shows raw intervals between decode timestamps, not
instruction execution times. COMPUTE/DMA tracks show inferred windows from a
UE instruction's decode to the next non-prefetchable instruction's decode,
skipping prefetched register/PBI operations (ISA types 2 and 6). These windows
include dispatch/fetch overhead and are not direct measurements of engine busy
or AXI bus utilization. An unavailable completion boundary produces no duration.
Only the final 8192 decode events are retained: single-bin execution does not
remove the trace BRAM limit. Missing prefixes are not painted as queue load.
The exporter checks the write pointer against the single-HALT program.

Trace is opt-in and adds register-read/export overhead to `execution_elapsed_s`.
Use `fpga_execution_s_sum` for hardware timing and run without trace for host
latency measurements. Reusing the same directory overwrites matching trace files.
Migration from the old layerwise format requires bin recompilation, not RTL changes.

Remaining work: labeled Carvana Dice/IoU, physical AXI256 validation,
more resolutions, CI integration and further device-side scheduling optimization.
