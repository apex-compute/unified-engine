# Prefill MLP: row-shard vs tensor-parallel, one layer on hardware

Prototype for moving the Omni map to **1 GiB private windows per core**. That map
only fits if the prefill MLP stops reading one shared copy of gate/up/down
(2889 MiB of the 3655 MiB decoder) and reads a private column shard instead.
This measures the cost of that change on one layer before converting 28.

Run with `models/qwen2.5_omni_7b/qwen2.5_omni_7b_mlp_tp_test.py`, 8 engines,
xdma0, 8 GiB, IF4 gate/up/down, H=3584, MLP=18944.

## The two paths

Transposed assignments of identical arithmetic:

| | row-shard (today) | tensor-parallel |
| :--- | :--- | :--- |
| engine `e` owns | rows `[e*M/8, …)`, all 8 down-K lanes | all `M` rows, lane `e` only |
| gate/up weights | one SHARED copy, lane by address arithmetic | private `[MLP/8, H]` shard |
| down weights | 8 shared pre-sliced lane blobs | private `[H, MLP/8]` shard |
| lane reduction | local accumulate into `[rows, H]` | cross-engine `reduce_add` |

The SiLU-multiply never leaves its lane in either path.

## Results

Latency is the HW counter for one layer. Run-to-run variance < 0.05%.

| rows `M` | row-shard | tensor-parallel | delta |
| ---: | ---: | ---: | :--- |
| 64 | 245.3 GFLOPS | **285.8** | **tp 14.2% faster** |
| 128 | 272.1 | **286.0** | tp 4.8% faster |
| 256 | 284.9 | **287.0** | tp 0.7% faster |
| 512 | **291.1** | 287.5 | tp 1.2% slower |

Peak is 307.2 GFLOPS (300 MHz x 128 FLOP/cycle x 8 engines).

**Tensor-parallel holds ~286 GFLOPS at every tile size** because each engine
always gets a full-height matmul. Row-shard degrades as the tile shrinks: at
M=64 each engine gets only 8 rows, and the 64-row-block kernels lose 20% of peak
to it. The two cross only above M=256, where row-shard's lack of a cross-engine
reduction finally pays off.

**The production operating point is M=64.** `_prefill_execution_rows()` rounds a
prompt to whole 64-row blocks GLOBALLY, so the 29-token prompt in the current
baseline runs M=64 -- exactly where tensor-parallel is 14.2% ahead. M=512 is only
reached by a 385+ token prompt.

## Other measured effects

- **Weight residency per engine:** 12.90 MiB private (tp) vs 169.20 MiB shared
  (row-shard) -- the whole point of the change.
- **Program size:** 550 instructions / 0.017 MiB (tp) vs 3198 / 0.098 MiB
  (row-shard), 5.8x smaller, because each engine emits one lane instead of eight.
- **Numerics:** identical SNR against a CPU reference built from the dequantized
  staged blocks -- 42.6 dB at M=64, 40.8 dB at M>=128, both paths.

The run also exercised the proposed 1 GiB arena and `PrivateArena.alloc_shared`
on real hardware, not only in simulation.

## Conclusion

Convert the prefill MLP to tensor-parallel. It is faster at the real operating
point, numerically identical, frees 2889 MiB of shared weights, and shrinks the
prefill program. The only regression is ~1% at the maximum 512-row tile.
