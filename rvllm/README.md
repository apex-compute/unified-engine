# RVLLM Support Branch

This branch is prepared for using RVLLM as the inference stack around the Apex
Compute Unified Engine FPGA software.

The bridge lives in the RVLLM checkout as `rvllm_apex`:

```bash
cd /Users/m5pro/git/solidsf/tools/rvllm
python3 -m rvllm_apex profiles
python3 -m rvllm_apex probe \
  --apex-repo /Users/m5pro/git/unified-engine \
  --dev xdma0 \
  --profile kintex7
```

The probe checks the XDMA device files, reads the Unified Engine hardware
version register, and refuses to continue when the bitstream or permissions are
wrong. The current expected public hardware hash is `0x253d5525`.

To launch one of the Apex model scripts through the RVLLM bridge:

```bash
cd /Users/m5pro/git/solidsf/tools/rvllm
python3 -m rvllm_apex run-model \
  --apex-repo /Users/m5pro/git/unified-engine \
  --dev xdma0 \
  --profile kintex7 \
  gemma3 -- --prompt "hello from RVLLM"
```

Use `model-command` to print the command without running it.

## Rust Runtime Scaffold

This branch also adds a Rust scaffold under `rust/apex-ue` for the pieces that
should not stay Python-only long term:

```bash
cargo run --manifest-path rust/apex-ue/Cargo.toml --bin apex-probe -- \
  --dev xdma0 --profile kintex7

cargo run --manifest-path rust/apex-ue/Cargo.toml --bin gemma4-e2b-plan -- \
  models/gemma4_e2b/gemma4_e2b_config.json
```

`apex-probe` mirrors the RVLLM fail-closed device probe in Rust. The Gemma 4
2B planner reads the repo's `models/gemma4_e2b/gemma4_e2b_config.json` and
reports the DRAM layout, KV-cache estimate, and SRAM tiling strategy without
requiring FPGA hardware.

For the current Gemma 4 2B feasibility notes, see
[`rvllm/GEMMA4_E2B.md`](GEMMA4_E2B.md).

## Hardware Profiles

The Apex repo currently exposes several board profiles in scripts:

| Profile | Part / family | Apex clock | AXI width | PCIe expectation |
|---|---:|---:|---:|---|
| `kintex7` | `xc7k480t` | 194 MHz | 256 bit | Gen2-class endpoint, up to x8 |
| `rk` | `xcku5p` | 333 MHz | 512 bit | verify with `lspci` |
| `puzhi` | `xcku5p` | 333 MHz | 256 bit | verify with `lspci` |
| `alveo` | `xcu50/u55n` | 250 MHz | 256 bit | Gen3 x16 or dual Gen4 x8 capable |
| `bittware` | Kintex UltraScale 15P class | 300 MHz | 512 bit | verify exact card with `lspci` |

None of the known Apex profiles are PCIe Gen5. On a Linux host with the card
installed, confirm the actual negotiated link:

```bash
lspci -D -nn | grep -i xilinx
lspci -s <BDF> -vv | grep -E 'LnkCap|LnkSta'
```

## Integration Shape

Apex's public stack is Python plus Xilinx XDMA and model-specific scripts. RVLLM
therefore owns the bridge, environment, probe, and fail-closed launch wrapper,
while Apex's `user_dma_core.py` remains the authority for the FPGA instruction
API.

The intended migration path is Rust-first around the stable seams: XDMA device
access, board/profile detection, manifest parsing, memory planning, and launch
orchestration. The large Python model files can then be reduced layer by layer
once the Rust instruction builders can be tested against real hardware.
