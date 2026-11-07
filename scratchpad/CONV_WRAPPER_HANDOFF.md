# Conv path wrapper: handoff (branch `winograd`, 2026-11-05)

## Goal
Every model currently lowers its convolutions by hand (im2col + matmul, CONV2D walker, or stereo_ess's Winograd).
We want ONE decision point: a thin wrapper above `user_dma_core` that looks at a conv's shape and weight type and
picks the lowering, so a 3x3 / stride 1 / dilation 1 / bf16 conv automatically takes Winograd F(2x2,3x3) and
everything else falls back to the right path without the model writer choosing.

## What is in the branch
- `user_dma_core.py`
  - `select_conv_path(...)` (module level, ~line 13600): 1x1 -> `matmul`; quantized weights -> `walker`;
    3x3/s1/d1/bf16/Cin >= `CONV3X3_WINOGRAD_MIN_CIN` (96) -> `winograd`; other bf16 -> `CONV3X3_BF16_FALLBACK`
    (`walker` when the weight pattern fits URAM B, else `im2col`).
  - `UnifiedEngine.conv2d_auto(...)` (~line 10056): runs the selector, emits `walker` itself via `conv2d_core`
    with `weight_reuse=True`, and calls the caller-supplied `winograd=` / `im2col=` / `matmul=` callables for the
    rest (those lowerings live in model code because their operands differ). `path=` is a per-layer override for
    layers that cannot afford Winograd's ~0-0.3 dB.
  - Two knobs: `CONV3X3_WINOGRAD_MIN_CIN = 96`, `CONV3X3_BF16_FALLBACK = "walker"`.
- `models/stereo_ess/utility/conv3x3_paths_probe.py`: board probe that times the four lowerings on the same
  shapes, same engines, same input, with SNR beside each:
  `direct` (im2col + matmul), `winograd` (SRAM transforms + 16 matmuls), `native16` (CONV2D, bf16 weights,
  weight_reuse), `native8` (CONV2D, IF8 weights, weight_reuse). Cases: dec3 96x160 192->64, enc1 48x80 64->64,
  dec1 24x40 640->256, dec0 12x20 768->512.

## The open question the probe answers
Winograd is expected to be fastest for its shape. The order of the OTHER two bf16 paths is unknown: the 927fd8d
RTL added weight reuse (the walker holds the conv's [oc][tap] pattern in URAM B instead of re-streaming it per
output pixel) and a FLOAT16 mode that only works with reuse, so native CONV2D is now a real bf16 competitor to
im2col for the first time. The probe's last block prints, per case, whether `native16` or `direct` is faster.

## Instructions for the next machine
1. **Have Rohit (the developer) run the probe for you first.** It touches the FPGA; do not run it yourself and
   do not run anything else on the device at the same time. Board must carry the 927fd8d image (main's
   `user_dma_core` asserts it).
   ```
   cd ~/unified-engine && conda run -n apex-compute --no-capture-output \
     python models/stereo_ess/utility/conv3x3_paths_probe.py --engines 8 2>&1 | tee scratchpad/conv3x3_paths_e8.log
   ```
   Then the same with `--engines 1`. `--cases dec3` gives a quick first signal; `--paths direct,winograd` isolates
   the native paths if they hang.
2. **Decide the fallback from the verdict lines.** If `native16` beats `direct` on the shapes that matter, keep
   `CONV3X3_BF16_FALLBACK = "walker"`. If `direct` wins, set it to `"im2col"` so the wrapper falls back to im2col
   before native conv. Also check `native16` SNR is >= 30 dB on every case; if it fails only there, the FLOAT16
   block packing in `conv2d_pack_weight_stream` is the first suspect.
3. Known first-run risks: the scheduler builds worker engines without the queue-config conv mode, so the probe sets
   `conv_geometry_mode` on them after construction. A hang or geometry assert in the native paths only points
   there. dec0 FLOAT16 uses 442 KB of the 512 KB URAM B per launch.
4. After the fallback is settled: lift stereo_ess's Winograd emitter (`_winograd*`, `_wino_fused`, `_sram_gather`
   on branch `stereo_models`, ~1,100 lines) into a shared module with a plain-layout DRAM-flow entry, validate it
   bit-identical against ess at 1 and 8 engines, then switch models to `conv2d_auto`. Rules that still hold:
   Winograd is bf16-only, needs the caller's accuracy veto (ffs's hourglass lost 13 dB), and only F(2,3) (larger
   tiles lose 10-12 dB in bf16).
