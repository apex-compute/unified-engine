# DPDFNet2 on Andromeda: Benchmark Guide

DPDFNet2 is a stateful streaming speech-enhancement model. Its pinned ONNX
graph consumes one `[1,1,161,2]` complex spectrum frame and a 45,424-element
state. Andromeda executes all 472 ONNX nodes from one resident single-bin; one
additional operation commits recurrent state. Runtime accounting must show no
CPU neural operations or intermediate host transfers.

## Reproduce remotely

Activate the desired Python environment in a clean Unified Engine checkout,
then run the one-file preparation flow:

```bash
bash models/dpdfnet/prepare_benchmark.sh
```

The script installs model dependencies, downloads and verifies the official
ONNX, compiles the single-bin, records repository revisions, and creates a
deterministic 100-frame input. If Andromeda is not a sibling checkout, set
`ANDROMEDA_REPO=/path/to/andromeda` before running it.

Run the Bittware AXI-256 benchmark separately:

```bash
python3 models/dpdfnet/dpdfnet_run_from_bin.py \
  --input perf_logs/dpdfnet_bittware_256/input_100_frames.npy \
  --output perf_logs/dpdfnet_bittware_256/output_100_frames.npy \
  --device bittware \
  --dev xdma0 \
  2>&1 | tee perf_logs/dpdfnet_bittware_256/run_100_frames.log
```

For a separate trace run, append:

```bash
--trace-tail perf_logs/dpdfnet_bittware_256
```

Run exactly the same input on an RK AXI-256 system (change `--dev` if its
XDMA device number differs):

```bash
mkdir -p perf_logs/dpdfnet_rk_256
cp perf_logs/dpdfnet_bittware_256/input_100_frames.npy \
  perf_logs/dpdfnet_rk_256/input_100_frames.npy
python3 models/dpdfnet/dpdfnet_run_from_bin.py \
  --input perf_logs/dpdfnet_rk_256/input_100_frames.npy \
  --output perf_logs/dpdfnet_rk_256/output_100_frames.npy \
  --device rk \
  --dev xdma0 \
  2>&1 | tee perf_logs/dpdfnet_rk_256/run_100_frames.log
```

Run the pinned ONNX graph on CPU using the identical spectrum frames. The
default is one CPU thread so that results are reproducible; record the CPU
model alongside the result. This is a reference benchmark only and is not
part of the hardware-only deployment path.

```bash
mkdir -p perf_logs/dpdfnet_cpu
python3 models/dpdfnet/dpdfnet_benchmark_cpu.py \
  --input perf_logs/dpdfnet_bittware_256/input_100_frames.npy \
  --output perf_logs/dpdfnet_cpu/output_100_frames.npy \
  --threads 1 \
  2>&1 | tee perf_logs/dpdfnet_cpu/run_100_frames.log
lscpu | tee perf_logs/dpdfnet_cpu/lscpu.txt
```

After each FPGA run, compare its saved output with the CPU reference:

```bash
python3 models/dpdfnet/dpdfnet_compare_outputs.py \
  --reference perf_logs/dpdfnet_cpu/output_100_frames.npy \
  --candidate perf_logs/dpdfnet_bittware_256/output_100_frames.npy

python3 models/dpdfnet/dpdfnet_compare_outputs.py \
  --reference perf_logs/dpdfnet_cpu/output_100_frames.npy \
  --candidate perf_logs/dpdfnet_rk_256/output_100_frames.npy
```

## Performance metrics

### Core performance comparison

All execution rows cover the same 100 consecutive streaming frames. FPGA
figures come from the cycle counter; CPU figures cover only ONNX inference.

| Metric | Bittware AXI-256 | RK AXI-256 | CPU (1 thread) |
|---|---:|---:|---:|
| **Execution time, 100 frames** | **2.984151 s** | — | **0.065881 s** |
| **Mean latency per frame** | **29.8415 ms** | — | **0.6588 ms** |
| **Throughput** | **33.51 FPS** | — | **1,517.88 FPS** |
| **Relative speed (Bittware = 1.00x)** | **1.00x** | — | **45.30x** |
| **Real-time factor (10 ms hop)** | **2.9842** | — | **0.0659** |
| Real-time streaming | No | — | Yes |

The current single-thread CPU is **45.30x faster** than Bittware AXI-256 by
device execution time. Equivalently, Bittware currently delivers about 2.21%
of this CPU's throughput. An RTF below 1.0 is required to keep up with the
model's 10 ms streaming hop.

### Platform and deployment overhead

| Metric | Bittware AXI-256 | RK AXI-256 | CPU / ONNX Runtime |
|---|---:|---:|---:|
| Device / processor | Bittware (`xdma0`) | — | Intel Core Ultra 9 285K |
| FPGA clock / CPU threads | 300 MHz (3.3333 ns) | — | 1 thread |
| Frames | 100 | — | 100 |
| Program instructions | 79,010 | — | N/A |
| Resident image / ONNX size | 18,997,184 bytes | — | 10,178,747 bytes |
| Artifact load / session creation | 0.158963 s | — | 0.042987 s |
| FPGA model upload | 0.006316 s | — | N/A |
| Host-observed graph execution | 3.062974 s | — | 0.065881 s |

### Correctness and hardware accounting

| Metric | Bittware AXI-256 | RK AXI-256 | CPU reference |
|---|---:|---:|---:|
| Input uploads | 100 | — | N/A |
| Program kicks / HALTs | 100 / 100 | — | N/A |
| Output reads | 100 | — | N/A |
| Intermediate host transfers | 0 uploads / 0 reads | — | N/A |
| Relative L2 error vs CPU | 0.411369 | — | Reference |
| RMSE vs CPU | 0.000165100 | — | Reference |
| Finite output | Yes | — | Yes |

For hardware, mean latency is `fpga_execution_s_sum / frames` and real-time
factor is `fpga_execution_s_sum / (frames * 0.010)`. Do not include artifact
loading, model upload, host orchestration, or trace export in core performance.

The populated Bittware and CPU columns were measured on 2026-09-11 using the
same deterministic 100-frame input. FPGA throughput, latency, and real-time
factor use the hardware cycle counter rather than host elapsed time. The CPU
column uses one ONNX Runtime intra-op thread; its image-size entry is the ONNX
file size, while the FPGA entry is the uploaded resident model/program image.
