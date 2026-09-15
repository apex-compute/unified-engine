# Numerical debugging evidence

The controlled tests identify precision loss in recurrent arithmetic and codebook score comparisons as major sources of error. They keep the official model structure and isolate specific numerical boundaries.

## Native rounding and compensation

Native elementwise arithmetic rounds through **BF19 with 10 fraction bits**, then rounds again to **BF16 with 7 fraction bits**. Direct BF16 CPU arithmetic can therefore predict different results and rank candidate fixes incorrectly.

The [native arithmetic proof](numerics_probe_execute.json) checks the emitted production helpers against this double-rounding model and independent FP32 results:

- All 133,120 product and sum high/low components match the native rounding model bit for bit. Recovered products equal the FP32 products of the original BF16 operands exactly within the tested finite ranges.
- Recovered sums remain approximate: 154 discrepancies, relative L2 `6.63e-7`, and maximum absolute error `0.0009765625`. A BF16 high/low pair is not general FP32 arithmetic.
- Sixty-four additions of `1/256` to a cell initially at `16` recover exactly `16.25` with compensation.
- Compensated Padé tanh has maximum absolute error `0.00221062` versus FP32 on 33,600 test elements, compared with `0.01516581` for the baseline. Quiet normal values through `2^-10` are exact; both signs of every normal BF16 code from `2^-126` through `32` are covered. Subnormal behavior is outside this contract.

Each case uploads its program/constants and guarded buffers, issues one START, receives one HALT, and reads outputs from DRAM. All outputs were finite; input buffers and guards were unchanged. The proof uses scalar DMA on Italy, build `0x40519e0a`, with the hardware lock, version, AXI256 and detected DRAM bounds checked before upload.

## Isolated decoder LSTM

The fix folds the two biases in FP32 before packing, combines recurrent projection and input projection before the sigmoid/BF16 gate writeback, and retains smaller terms through compensated cell and tanh arithmetic.

The [fresh production LSTM proof](lstm_probe/native.json) regenerates its input from the official checkpoint and **317 fixed FPGA-selected token IDs**. Both kernels receive the identical BF16 input; the CPU reference uses the original FP32 input. No encoder or token selection is rerun. Both native results then pass through the same official FP32 decoder tail.

Local LSTM relative L2 error falls from **2.1353% to 0.9541%**, and the subsequent FP32-tail waveform error falls from **23.5681% to 10.4768%**. Current production outputs match the earlier measured outputs bit for bit. These are isolated decoder controls; the [LSTM report](lstm_probe/README.md) gives input/reference hashes, runtime, scope and reproduction commands.

## Codebook score ties

Normalized candidate dot products near one can collapse to the same BF16 score. For example, scores near `0.934` have spacing `0.00390625`. Subtracting the same constant from all candidates preserves the mathematical argmax. The optional score-centering path applies `−1` inside the native score dot operation, before BF16 writeback; scores near `−0.066` then have eight times finer spacing.

The optional compensated codebook stores each FP32 unit-codebook component as a BF16 high part and a BF16 residual in unused lanes, and duplicates the query into those lanes. Query normalization still uses the original eight dimensions. This preserves more codebook information while retaining the existing 64-lane dot geometry.

Centering cannot recover distinctions already lost in the native reduction tree. Equal remaining scores use the stable lower-ID tie rule. The [score-tree comparison](short_clip/records/vq_tree_comparison.json) translates older local RTL to CPU and checks its predictions against saved native token IDs. The [short-clip evidence](short_clip/README.md) records the exact configurations and the limits of that comparison.

The remaining isolated LSTM and waveform errors above are material. Encoder numerical changes can also alter discrete token choices, so improvements in an isolated decoder do not by themselves establish improved end-to-end waveform accuracy. Recording-level comparisons are reported separately.

## Reproduce the primitive proof

With the [BigCodec dependencies](../../requirements.txt) installed, run from the repository root:

```sh
# Offline capture and CPU contracts; choose an available core on another host.
python models/bigcodec/validation/20260915_accuracy/numerics_probe.py --cpu-core 11

# Explicit native execution on Italy.
python models/bigcodec/validation/20260915_accuracy/numerics_probe.py --execute --cpu-core 6
```

The utility needs no checkpoint or ignored diagnostic files. Reports include source hashes and separate FP32 and native-rounding contracts. Only compact Python, Markdown, JSON and text logs are packaged for these primitive and LSTM proofs; captured programs and large tensors are not saved.
