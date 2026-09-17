# Qwen2.5-Omni: 1 GiB per core, tensor-parallel prefill

The map now gives every core a **1 GiB private window**, with its ISA and tensor
scratch inside it, and the windows tile the whole 8 GiB device. Shared data lives
in the empty tail of those windows. The prefill MLP is tensor-parallel, so 79% of
the decoder weights are private and never duplicated.

## The map

```
core i -> [i GiB, (i+1) GiB), low to high inside each window:
  weights   1000 MiB   private shards, bump-allocated UP from the base
  (gap)                the shared pool, bump-allocated DOWN from the top
  ISA         16 MiB
  tensor       8 MiB   per-engine scratch
```

Because the windows span the whole device there is no region left above them, so
shared data is carved from the gaps and `PrivateArena` arbitrates the two cursors.
Private space is declared up front (`reserve_private`) so the pool can never lend
away space a later private shard needs.

Per core: **707.5 MiB private** (audited against a 712 MiB reserve at load),
leaving ~288 MiB/core of pool. Peak shared use is ~1053 MiB against ~2240 MiB.

Two objects cannot be scattered and get dedicated extents carved at init, before
anything competes: the **256 MiB tensor extent** (activations + KV, addressed as
whole `[M, N]` buffers) and the **280 MiB untied head** (one contiguous 276 MiB
IF4 blob). Everything else places section by section: 196 attention sections
(max 6.5 MiB), 455 vision, 489 audio.

## Weight duplication: what is shared and what is not

| | bytes | where |
| :--- | ---: | :--- |
| MLP gate/up (N-shard, serves BOTH phases) | 240.8 MiB/core | private |
| MLP down, N-shard (decode) | 120.4 MiB/core | private |
| MLP down, K-shard (prefill TP) | 120.4 MiB/core | private |
| attention, decode shard | 38.3 MiB/core | private |
| lm_head shard, embedding, BF16 O | 187.6 MiB/core | private |
| attention + head + norms (prefill) | 765 MiB | shared pool |
| vision / audio | 729 MiB | shared pool, time-shared |

**2889 MiB of 3655 MiB (79%) of the decoder is private with zero duplication.**
`gate`/`up` are the payoff: prefill's tensor-parallel lane and decode's column
block want the SAME N-shard, so one copy serves both phases.

`down` is the one weight stored twice, and it is unavoidable: prefill must split
the CONTRACTION dim to match gate/up's column split, decode splits the OUTPUT dim
so its M=1 result concatenates. A K-slice is strided, so it cannot be re-derived
from the N-slice by address arithmetic. Duplicating gate/up instead of down would
NOT fit (948 MiB private leaves 286 MiB of pool against a 1053 MiB peak).

Nothing is ever re-quantized. The IF4 image is `[N, K/64]` bf16 scales then
`[N, K/2]` nibbles, so an N-slice is a contiguous byte range and a K-slice a
fixed byte window of every row -- exact in both directions, because the lane is
2368 = 37 whole scale blocks and an even number of nibbles.

## Tensor-parallel prefill

Engine *e* runs ALL rows for lane *e*, instead of its own rows for all lanes.
Four regions per layer: residual+norm (row-sharded, joins), gate/up/multiply/down
(column-sharded, stays in lane), one `reduce_add`, residual2.

Measured, 8 engines on xdma0:

| | baseline | now | |
| :--- | ---: | ---: | :--- |
| Prefill, 29 tokens (text) | 227.3 GFLOPS | **263.5** | +16% |
| Prefill, 170 tokens (image) | 97.6 GFLOPS | **275.4** | **+182%** |
| Vision encoder | 229.0 GFLOPS | 230.1 | unchanged |
| Prefill program | 6.60 MiB | 4.28 MiB | 1.5x smaller |

The 170-token case is where it shows: row-sharding gave each engine 24 rows, far
below the 64-row block the kernels are built for, and tensor-parallel gives every
engine a full-height matmul regardless of prompt length. A single-layer A/B is in
`qwen2.5_omni_7b_mlp_tp_prototype.md`.

Both modes produce correct output (text: the Rayleigh-scattering answer; image: a
correct description of the Yosemite sample).

## Decode: blocked on HBM configuration, not on this map

Decode measures 77-79 GFLOPS against a 140-145 GFLOPS baseline. The cause is a
DRAM address-range effect, not the map geometry or the sharding:

| layout (M=1, decode shape) | GFLOPS |
| :--- | ---: |
| 512 MB windows, low 4 GiB | 135.1 |
| 512 MB windows, upper 4 GiB | 64.3 |
| 1 GiB windows (spans 8 GiB) | 77.0 |

The upper 4 GiB runs at about half the bandwidth of the lower 4 GiB. The old map
kept all eight 512 MB private windows inside `[0, 4 GiB)` -- exactly the fast
half -- and put shared prefill weights above. Any 1 GiB-per-core map necessarily
places half the cores' data above 4 GiB, so every region's barrier waits on the
slow half. **This is an HBM feature that is not configured correctly on this
board, not a property of the design**; the same shards in the low 4 GiB run at
135 GFLOPS. Re-measure decode once HBM is configured.

## Files

- `multi_engine_shard.py` -- `alloc_shared`, `shared_mark`/`shared_release`,
  `reserve_private`. Backward compatible: 512 MB geometry, bounds and error text
  are unchanged when `alloc_shared` is never called, so Gemma3/Gemma4/Qwen-VL are
  untouched.
- `qwen2.5_vl_3b_lm.py` -- default-off hooks (`_lm_projection_is_private`,
  `_decode_shard_override`, `_prefill_mlp_tp_engines`) plus the generic TP
  emitter. The VL model's behaviour is unchanged.
- `qwen2.5_omni_7b_test.py` -- the map, the params-to-pool redirect, reserved
  extents.
- `qwen2.5_omni_7b_lm.py` -- private MLP staging, private BF16 O stripes, the
  reserve audit.
- `qwen2.5_omni_7b_weights.py` -- `_config_fingerprint` narrowed to an explicit
  allowlist. It used to hash the whole config, so a DRAM-map edit invalidated
  5.9 GB of bit-identical weights; `hardware` provably does not affect params
  content. Proven map-independent before migrating the stored hash.
