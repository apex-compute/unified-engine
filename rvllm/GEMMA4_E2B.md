# Gemma 4 E2B on Apex Unified Engine

Target model path: `models/gemma4_e2b`.

The repo already contains a Gemma 4 E2B implementation with a fixed
multimodal layout:

| Region | Range | Size | Use |
|---|---:|---:|---|
| Weight LM | `0x00000000..0x64000000` | 1600 MiB | language weights and embeddings |
| Weight Vision | `0x64000000..0x6c000000` | 128 MiB | vision weights |
| Weight Audio | `0x6c000000..0x78000000` | 192 MiB | audio weights |
| Activation Scratch | `0x78000000..0x88000000` | 256 MiB | stage scratch |
| Activation KV | `0x88000000..0x98000000` | 256 MiB | K/V cache |
| ISA Audio | `0x98000000..0xa0000000` | 128 MiB | audio program region |
| ISA Unified | `0xa0000000..0x100000000` | 1536 MiB | unified programs.bin |

Config-level feasibility looks good for the current quantized layout:

- Layers: 35
- Hidden size: 1536
- Vocab: 262144
- Group size: 8
- Max context: 1024
- Prefill cap: 512
- Sliding window: 512
- Full/global attention layers: 4, 9, 14, 19, 24, 29, 34
- Q bytes per token: 8192
- K bytes per token: 1024
- Max config file offset: `0x9046bd00`. This is a source-bin high-water mark,
  not direct DRAM residency.
- Runtime params DRAM usage: `0x6046dd00`, about 1540 MiB, leaving about
  59.6 MiB headroom in the 1600 MiB params region.

The Python runner extracts the exact KV-sharing map from the HF model manifest.
Using the public code comments as a config-only estimate, layers 15-34 share KV
state, so the runtime needs about 15 unique KV slots instead of 35. That is
about 30 MiB resident K+V cache at 1024 context, saving about 40 MiB against a
no-sharing layout.

## SRAM Strategy

Use SRAM/URAM as a streaming tile store, not as model storage. The public
hardware exposes two 512 KiB sections, about 1.0 MiB total:

- SRAM A: operand/result tile
- SRAM B: second operand or streamed matrix tile
- Safe chunk used by the Gemma 4 code: 131072 bf16 elements, 256 KiB

At Gemma 4 E2B dimensions, that safe chunk covers about 85 hidden rows
(`1536` bf16 elements each) or about 10 wide-MLP rows (`12288` bf16 elements
each). Whole layers and the K/V cache should remain in DRAM. The profitable
pattern is to stream Q/K/V, RMSNorm, residual, and MLP tiles through SRAM while
keeping instruction and weight layout deterministic.

## Current Run Path

The current Apex implementation can only be claimed "possible" after probing a
real Linux host with the card installed:

```bash
cargo run --manifest-path rust/apex-ue/Cargo.toml --bin apex-probe -- \
  --dev xdma0 --profile kintex7

cargo run --manifest-path rust/apex-ue/Cargo.toml --bin gemma4-e2b-plan -- \
  models/gemma4_e2b/gemma4_e2b_config.json
```

The Mac used to prepare this branch has no `/dev/xdma*` nodes, so inference
launch is intentionally blocked here.
