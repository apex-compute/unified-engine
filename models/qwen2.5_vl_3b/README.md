# Qwen2.5-VL-3B

Vision-Language Model (VLM) inference on the Apex Compute Unified Engine.
Vision encoder + LM prefill + decode, single core or up to 8 cores.

## Layout

- **qwen2.5_vl_3b_test.py** – entrypoint: CLI, DRAM map, engine class, VLM flow
- **qwen2.5_vl_3b_vision.py** – `Qwen25VLVisionMixin`: encoder weight init, compile, run
- **qwen2.5_vl_3b_lm.py** – `Qwen25VLLMMixin`: LM weights, prefill, decode, sharding
- **qwen2.5_vl_3b_numeric.py** – numeric debug: FPGA vs host-simulated IF4 vs HuggingFace
- **qwen2.5_vl_3b_config.json** – model dimensions, precision, and paths
- **qwen2.5_vl_3b_bin/** – `params.bin` + `params.json` and the HF checkpoint
- **qwen2.5_vl_3b_model_graph.md** – hardware-agnostic compute graph (shapes, params, DRAM)

Structured after `gemma4_e2b`: a thin entrypoint plus one mixin per stage.

## Architecture

- **LM:** 36 layers, hidden 2048, GQA 16 Q / 2 KV heads (group 8), head_dim 128,
  SwiGLU (intermediate 11008), Q/K/V bias, no QK-norm, mRoPE `[16,24,24]`,
  tied LM head, vocab 151936. IF4 weights; V and O kept BF16 for attention accuracy.
- **Vision encoder:** 32 layers, hidden 1280, 16 heads, head_dim 80 (padded to
  128 on-chip), SwiGLU (intermediate 3420), RMSNorm, window attention
  (112-px windows) with full attention at layers 7/15/23/31; 2×2 patch merger
  → 144 tokens × 2048. IF4 weights.

## Usage

```bash
# LM / text-only, default prompt
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py

# VLM: default sample image (test_samples/yosemite.jpg) + "Describe the picture in details."
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image

# VLM with a custom image and prompt
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image /path/to/photo.jpg --prompt "What do you see?"

# 8 cores, with the per-phase profile and a run report
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py --image --multi-core 8 --profile
```

Args follow gemma4: `--prompt --image [PATH] --max-new-tokens --profile --profile-ctx
--multi-core [N] --dev`. Vision runs when `--image` is given; otherwise
text-only. `--profile` additionally writes a markdown run report next to the script,
named for the args used (e.g. `qwen2.5_vl_3b_test_xdma0_image_multi-core_8_profile.md`).

## Performance

FPGA (`xdma0`, 366.7 MHz), image + 170-token prompt, greedy decode. Decoded text is
identical at 1 and 8 cores.

| phase | 1 core | 8 cores | speedup |
| :--- | ---: | ---: | ---: |
| Vision (576 patches → 144 tokens) | 21990.2 ms | 3306.2 ms | 6.65× |
| Prefill (170 tokens) | 22135.0 ms | 3185.5 ms | 6.95× |
| Time to first token (TTFT, CPU; vision + prefill) | 44.13 s | 6.50 s | 6.79× |
| Time to first token (TTFT, HW counter; vision + prefill) | 44.13 s | 6.49 s | 6.80× |
| Decode, 1st token | 207.9 ms (4.81 tok/s) | 33.3 ms (30.02 tok/s) | 6.24× |
| Decode, average | 214.0 ms (4.67 tok/s) | 34.3 ms (29.13 tok/s) | 6.24× |
| Decode, long ctx (2048 tokens) † | 282.2 ms | 51.0 ms | 5.54× |

† From `--profile`, which times one decode step at a full 2048-token context in
addition to the live one. It is where attention dominates: 84.5 ms of the 1-core
step, and the phase that the decode attention shard targets.

See `pr_summary.md` for the per-phase breakdown.

## DRAM map

Upper 2 GB (identical at every core count):

```
PARAMS  0x8000_0000 - 0xF100_0000   1808 MiB   time-shared: vision (389.7 MiB)
                                               loads, runs, then LM (1801.7 MiB)
                                               overwrites the same addresses
TENSOR  0xF100_0000 - 0xFB00_0000    160 MiB   activations + KV cache (122.8 used)
ISA     0xFB00_0000 - 0x1_0000_0000   80 MiB   master 24 MiB, then one slice per worker
```

Multi-core private space is the whole lower 2 GB, one window per engine laid out
`[ weights | tensor ]` — 8 engines → 256 MiB/window (240 MiB weights + 16 MiB
tensor). Worker ISA lives in the model map above, not in the windows.

## Multi-core sharding

| stage | sharded | on the master |
| :--- | :--- | :--- |
| Vision | qkv_proj, attention (per head), o_proj+mlp, rope | permutes, merger |
| Prefill | qkv, o_proj, mlp, attention (per head), norms/eltwise | rope |
| Decode | q/k/v, o_proj, gate/up + SwiGLU product, down + residual, lm_head, attention (V transpose + P@V^T) | rope, Q@K^T, permute |

Decode weights are duplicated into each engine's private arena (column shards) so
engines do not contend on one shared weight image; 228.4 MiB of the 240 MiB window
is used at 8 cores. Ops too narrow to reach every engine shard over a subset
(k and v are N=256 → 4 engines of 8). RoPE is unsharded pending library support.

The largest remaining serial op is **decode attention**, which grows with context:
73.6% of the 8-core step at ctx 2048. Batch-splitting it is the next win.

## Numeric verification

```bash
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_numeric.py --image          # vision
python models/qwen2.5_vl_3b/qwen2.5_vl_3b_numeric.py --lm             # LM prefill
```

Two references per run: a host simulation using the same IF4 weights (a low SNR
here is an emission bug) and HuggingFace bf16 (the remaining gap is quantization).
Vision measures 18.7 dB encoder / 21.4 dB merged against the IF4 host reference;
LM prefill 28.3 dB.
