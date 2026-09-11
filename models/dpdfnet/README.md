# DPDFNet2 on Andromeda

This directory starts the 16-kHz `dpdfnet2` port from CEVA's Apache-2.0
DPDFNet repository. The source of truth is the official streaming ONNX model,
pinned by SHA256. It consumes one complex spectrum frame represented as real
FP32 `[1, 1, 161, 2]` plus a flat 45,424-element recurrent state, and returns
one enhanced frame plus the next state. STFT and iSTFT are intentionally host
operations and are not present in the ONNX graph.

The deployment contract is hardware-only for the complete ONNX neural graph:
the compiler may use ONNX/PyTorch while building the artifact, but the eventual
`run_from_bin` path may not invoke ONNX Runtime or PyTorch neural operators, and performs no
host-side intermediate tensor transfers beyond boundary packing. One spectrum frame is uploaded,
one resident FPGA program is kicked, recurrent state remains in device DRAM,
and the enhanced frame is read after the program's sole HALT. Audio STFT/iSTFT
is outside the supplied ONNX ABI; a strict zero-CPU application must provide
and consume spectrum frames outside this runner.

Current status is **end-to-end, stateful, single-bin hardware inference** for
the complete 472-node ONNX graph. The deployment image contains 473 operation
ranges: one per ONNX node plus the final recurrent-state commit. The runtime
uses no CPU neural-network operators and performs no intermediate host tensor
transfers. The model image is uploaded once; each spectrum frame uses one input
write, one program kick, one HALT and one output read.

The checked-in source supports AXI-256 and AXI-512 devices. The generated ONNX
and `dpdfnet2-andromeda.bin` files remain local build products under the ignored
`dpdfnet_bin/` directory.

Install only the model-specific inspection/reference dependencies:

```bash
python -m pip install -r models/dpdfnet/requirements.txt
```

Download, verify and inspect the official graph:

```bash
python models/dpdfnet/dpdfnet_prepare.py --download \
  --audit-json perf_logs/dpdfnet/dpdfnet2_operator_audit.json
```

The model is written below the ignored `models/dpdfnet/dpdfnet_bin/` directory
and must not be committed. Run the CPU streaming reference on a WAV. Input is
converted to 16 kHz for the model and converted back to its original sample
rate for output:

```bash
python models/dpdfnet/dpdfnet_run_cpu.py --download \
  --input path/to/noisy_16khz.wav \
  --output models/dpdfnet/dpdfnet_bin/enhanced.wav \
  --attn-limit-db 12
```

Compile the pinned ONNX graph into one stateful Andromeda image:

```bash
python models/dpdfnet/dpdfnet_compile.py --force
```

Run one or more precomputed spectrum frames entirely through the FPGA graph:

```bash
python models/dpdfnet/dpdfnet_run_from_bin.py \
  --input /tmp/dpdfnet_apex_16.npy \
  --output /tmp/dpdfnet_apex_16_hw.npy \
  --device bittware_512 --dev xdma0
```

Export the final frame's chronological 8192-event trace tail:

```bash
python models/dpdfnet/dpdfnet_run_from_bin.py \
  --input /tmp/dpdfnet_apex_16.npy \
  --output /tmp/dpdfnet_apex_16_hw.npy \
  --device bittware_512 --dev xdma0 \
  --trace-tail perf_logs/dpdfnet
```

Run dependency-free helper tests from the repository root:

```bash
python -m unittest discover -s models/dpdfnet -p 'test_*.py'
```

Verify the first supported dense ONNX Conv against a quantized CPU reference
on live hardware. The script exposes that node's real activation from an
in-memory ONNX graph, automatically selects gather-IF8, and exercises its
height-zero/width-one padding through CONV2D:

```bash
python models/dpdfnet/dpdfnet_conv_smoke.py \
  --device bittware_512 --dev xdma0
```

Add `--all` to validate all 15 dense Conv nodes using their actual one-frame
ONNX intermediates. This is a layer-by-layer hardware conformance suite, not
a whole-graph performance benchmark.

The 13 depthwise Conv nodes have a separate hardware-only tap-wise emitter and
conformance test. ONNX Runtime supplies only the oracle inputs/outputs for this
test; it is not part of the eventual deployment runtime:

```bash
python models/dpdfnet/dpdfnet_depthwise_smoke.py \
  --device bittware_512 --dev xdma0
```

Validate all 22 Gemm and 14 MatMul nodes, including the batched projection
weights used by the squeezed linear layers:

```bash
python models/dpdfnet/dpdfnet_linear_smoke.py \
  --device bittware_512 --dev xdma0
```

Validate all eight last-axis LayerNormalization nodes. The test reports the
small numerical effect of the current hardware primitive omitting ONNX's
`epsilon=1e-5` separately from its hardware-vs-BF16 error:

```bash
python models/dpdfnet/dpdfnet_layernorm_smoke.py \
  --device bittware_512 --dev xdma0
```

Validate all 20 Relu, 19 Sigmoid, and 10 Tanh nodes using the accelerator's
LALU activation paths (Tanh is lowered as `2*sigmoid(2*x)-1`):

```bash
python models/dpdfnet/dpdfnet_activation_smoke.py \
  --device bittware_512 --dev xdma0
```

Validate all Add/Mul/Sub nodes. ONNX broadcasts are expanded when the artifact
is prepared, so runtime execution remains a plain FPGA elementwise operation:

```bash
python models/dpdfnet/dpdfnet_eltwise_smoke.py \
  --device bittware_512 --dev xdma0
```

On the local AXI-512 Bittware image, a 16-frame validation run completed in
about 438 ms of FPGA time (about 27.4 ms/frame). Against the saved FP32 ONNX
reference sequence it produced an aggregate relative L2 error of about 0.17
and no NaN or infinity. These figures are validation baselines, not a portable
performance guarantee.
