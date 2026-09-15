# Isolated decoder LSTM comparison

With the same fixed input, the current production fused gates and compensated cell/tanh arithmetic reduced decoder LSTM error by **55.32%**. Passing its output through the official FP32 decoder tail reduced waveform error by **55.55%**. Both native outputs reproduced the earlier experimental outputs bit for bit.

| Measured native kernel | LSTM relative L2 error | FP32-tail waveform error | FPGA LSTM time | Instruction bytes |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.1353% | 23.5681% | 1.842046 s | 8,129,216 |
| Fused gates + compensated cell/tanh | 0.9541% | 10.4768% | 1.949770 s | 38,422,272 |

These [native production measurements](native.json) used build `0x40519e0a` on Italy at 333.25 MHz. Both recorded one START and one HALT, finite outputs, unchanged inputs and intact guards. The improved kernel took 5.85% longer for this isolated operator. Times exclude other model operators and host transfers; they are not whole-model real-time ratios.

The input comes from the **same frozen 317 FPGA-selected token IDs** for `p232_007`. The official FP32 `vq2emb` and first decoder convolution produce a `[317,1536]` tensor. Both native kernels receive its identical BF16 conversion and use BF16 recurrent weights. The reference uses the original FP32 tensor through the official two-layer `ResLSTM`, including its skip connection; all initial recurrent states are zero.

Each native result then passes through the unchanged official FP32 decoder tail. Waveform error uses the first 63,295 samples at 16 kHz (3.9559375 seconds), without resampling, alignment, gain or polarity fitting. The encoder and token selection are not rerun. This control isolates decoder recurrent arithmetic; it does not establish whole-model accuracy or encoder token agreement. The remaining waveform error is 10.48%.

An additional CPU control rounds only the LSTM input to BF16 and leaves the LSTM and tail in FP32. Its errors are 0.1236% locally and 0.3697% at the waveform, substantially below the native errors above.

The [independent historical review](historical_review.json) regenerates the official references directly from the pinned checkpoint and [committed token file](../../20260914_optimized_italy/p232_007_optimized_bf16_fpga.tokens.npz), then verifies the saved native output hashes and recomputes both errors. No ignored trace or diagnostic implementation is imported. The report contains the original execution records and hashes; the large native tensors are not copied here.

The [production capture](offline_capture.json) verifies the same input/reference hashes and captures the current baseline and fused kernels. The baseline instruction hash matches the historical kernel exactly. The current fused kernel has the same instruction count but a different instruction hash from the experimental kernel; the [fresh native replay](native.json) confirms identical output hashes for both variants. The baseline output is `c5d36036…c78b6ff`, and the improved output is `d1a820c0…6b8bef9d`; full hashes and source hashes are in the reports.

## Reproduce

From the repository root, install the documented dependencies and fetch the pinned checkpoint if needed:

```sh
python -m pip install -r models/bigcodec/requirements.txt
python models/bigcodec/bigcodec_fetch.py
```

Regenerate the references and capture both current kernels without accessing a device. Select an available CPU core when running on another host:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python models/bigcodec/validation/20260915_accuracy/lstm_probe/reproduce.py --cpu-core 11
```

On Italy, explicitly execute both kernels and compare their outputs through the FP32 tail:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python models/bigcodec/validation/20260915_accuracy/lstm_probe/reproduce.py \
  --execute --cpu-core 6 --expected-version 0x40519e0a
```

The script opens the existing hardware lock read-only, checks the stamped version, AXI256 and detected DRAM bounds, and uses scalar DMA. Each kernel uploads its constants, program and guarded input/workspace, receives one execution command, and returns its output from DRAM. CPU tail comparisons run after releasing the hardware lock. Only JSON is saved; no program bins or tensors are written. `--review-native BASELINE.pt IMPROVED.pt` can independently review previously saved native output/record pairs.

The primitive [native numerical proof](../numerics_probe_execute.json) also passed: 133,120 products recovered FP32 exactly, all product/sum components matched the BF19→BF16 rounding model bit for bit, and the small cell-update test recovered 16.25 exactly. Approximate sum residuals retained 154 discrepancies, with relative L2 `6.63e-7` and maximum absolute error `0.0009765625`. The 33,600-element tanh grid passed its independent FP32 accuracy contract, including every normal BF16 value from `2^-126` through `32` in both signs; subnormals are explicitly excluded.
