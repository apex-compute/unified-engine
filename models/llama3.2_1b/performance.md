# LLaMA 3.2 1B Performance

Baseline numbers captured from the per-run summary written by
`llama3.2_1b_test.py` (IF4). The run compiled the program image **from scratch**
and decoded **pure-greedy** (deterministic, matches `user_hw_test.py`).

- **HW version:** `0x4a6cf588`
- **--dev / --device:** `xdma1` / `kintex7`
- **Clock / frequency:** 5.0422 ns (198.3 MHz)
- **Cores:** 1
- **Prompt:** `default_prefill_tokens` (`x+3=5, what is x?`), 44 prefill tokens
- **Architecture:** 16 layers, hidden 2048, GQA 8 KV heads × 4 Q/KV = 32 Q heads,
  actual_head_dim 64, MLP 8192, IF4 quantization.
- **Source summaries:** the run now auto-writes a per-run Markdown summary
  (`write_run_summary`, same format as gemma3) named for the CLI config, e.g.
  [llama3.2_1b_test_xdma1_kintex7_puregreedy.md](llama3.2_1b_test_xdma1_kintex7_puregreedy.md);
  see also [run.md](run.md) for the FLOPs/scratch audit.

> **⚠️ Discrepancy vs the previous performance.md (2025-06-22, HW `0x25e4082c`,
> 5.15 ns).** The instruction image is now **~2.7–3× larger**: prefill
> **842.2 → 2,517.1 kB**, decoder **666.4 → 1,617.6 kB**, combined
> **1,508.7 → 4,134.6 kB**. This is expected: the current build **inlines**
> `unified_attention_core` once per KV head (8×) inside every layer body (16×),
> replacing the old shared `flash_attention` subroutine + `JUMP_ABS` call-site
> pattern — it trades image size for removing per-call dispatch. Timing improved
> on the newer bitstream/clock: prefill HW **3,944 → 3,664 ms**, decode average
> **~7.67 → 8.15 tok/s**. (The old and new rows use different HW versions and
> clock periods, so treat the timing deltas as indicative, not controlled.)

## LLaMA 3.2 1B IF4

| Metric | Baseline | proper-gqa |
|---|---:|---:|
| Peak throughput | 25.39 GFLOPS | 25.39 GFLOPS |
| Prefill program | 2,517.1 kB | 6,156.1 kB |
| Decoder program | 1,617.6 kB | 1,617.6 kB |
| Combined program image | 4,134.6 kB | 7,773.6 kB |
| Prefill tokens (seq_len) | 44 | 44 |
| Prefill FPGA time (HW latency) | 3,664.0 ms | 3,629.4 ms |
| Prefill throughput | - GFLOPS | 23.71 GFLOPS |
| — utilization (% peak) | - | 93.4% |
| Prefill end-to-end (CPU) | 3.66 s | 3.63 s |
| Decoded tokens | 62 generated (total 106) | 62 generated (total 106) |
| Decode 1st-token speed (peak) | 8.47 tok/s | 8.47 tok/s |
| Decode average speed | 8.15 tok/s | 8.20 tok/s |
| Decode 1st-token HW latency | 118.1 ms/tok | 118.1 ms/tok |
| Decode average HW latency | 120.6 ms/tok | 120.6 ms/tok |
| Decode average throughput | 21.65 GFLOPS | 21.65 GFLOPS |
| — utilization (% peak) | 85.3% | 85.3% |
| Correctness | coherent (solves x = 2) | coherent (solves x = 2, identical text) |

> **proper-gqa** (IF4, xdma1/kintex7, HW `0x4a6cf588`, pure-greedy, 44-token
> prompt): prefill attention rewritten from *duplicate-and-batch* (per KV head:
> replicate K/V `group_size`× and one `[4S,4S]` SDPA) to true GQA — an outer loop
> over the 8 KV heads × inner loop over the 4 query heads, each a compact `[S,S]`
> SDPA reading the un-duplicated K/V straight from the cache (plain causal bias).
> That's `group_size`(=4)× less attention MAC + KV DMA. **Bundled with the FLOP
> accounting fix**: prefill FLOPs are now computed at the *real* prompt length
> (projections `M = prompt_seq_len`; attention counted at the real compact dims),
> so the reported prefill GFLOPS reflects the actual work — which is *why the
> number drops vs Baseline (24.21 → 23.71): Baseline over-counted the duplicated
> attention*. Instructions stay fully seq_len-agnostic (all runtime loops GPR-driven;
> the attention core's static `batch`/`aligned` are pinned to `PREFILL_CONTEXT_SIZE`
> for scratch, never the prompt).
>
> **Verdict at 44 tokens: not worth it.** Prefill latency improves only **~1%
> (3,664 → 3,629 ms)** because at this length attention is a tiny slice of prefill
> (projections/MLP dominate), while the prefill image **grows 2.4× (2,517 → 6,156
> kB)** from inlining 32 attention call-sites per layer instead of 8. Output is
> byte-identical (llama's group-aware duplicate bias was already numerically exact,
> so no token drift). The `O(seq²)` compute win only turns net-positive at long
> context; for short prompts the duplicate-and-batch baseline is the better trade.

## LLaMA 3.2 1B IF8

Same proper-GQA + real-seq_len FLOP change applied to `llama3.2_1b_IF8.py`
(xdma1/kintex7, pure-greedy, 44-token prompt). Summary:
[llama3.2_1b_IF8_test_xdma1_kintex7_puregreedy.md](llama3.2_1b_IF8_test_xdma1_kintex7_puregreedy.md).

| Metric | Baseline | proper-gqa |
|---|---:|---:|
| Peak throughput | 25.39 GFLOPS | 25.39 GFLOPS |
| Prefill program | 2,517.1 kB | 6,156.1 kB |
| Decoder program | 1,617.6 kB | 1,617.6 kB |
| Combined program image | 4,134.6 kB | 7,773.6 kB |
| Prefill tokens (seq_len) | 44 | 44 |
| Prefill FPGA time (HW latency) | 7,037.2 ms | 7,002.7 ms |
| Prefill throughput | - GFLOPS | 12.29 GFLOPS |
| — utilization (% peak) | - | 48.4% |
| Decoded tokens | 68 generated (total 112) | 68 generated (total 112) |
| Decode average speed | 4.57 tok/s | 4.57 tok/s |
| Decode 1st-token HW latency | 215.3 ms/tok | 215.3 ms/tok |
| Decode average HW latency | 217.9 ms/tok | 217.9 ms/tok |
| Decode average throughput | 11.98 GFLOPS | 11.98 GFLOPS |
| — utilization (% peak) | 47.2% | 47.2% |
| Correctness | coherent (solves x = 2) | coherent (solves x = 2, identical text) |

> **Same verdict as IF4, and same not-worth-it at 44 tokens.** Prefill latency
> moves **~0.5% (7,037 → 7,003 ms)** while the prefill image grows the identical
> **2.4× (2,517 → 6,156 kB)** — the prefill *instruction* structure is the same for
> IF4 and IF8 (only weight quantization differs), so proper-GQA's image cost is
> identical; IF8 is just ~2× slower per token from the 8-bit weight traffic.
> Decode is untouched (68 tokens, byte-identical text — matches the IF8 golden in
> `user_hw_test.py`, so no golden change). Reported prefill GFLOPS drops
> (12.60 → 12.29) for the same reason as IF4: the FLOP fix now counts the real
> compact attention instead of the duplicated work.
