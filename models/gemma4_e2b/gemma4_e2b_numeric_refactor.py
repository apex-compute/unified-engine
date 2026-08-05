#!/usr/bin/env python3
"""Vision-encoder numeric check for the REFACTOR engine (gemma4_e2b_refactor.py).

Reuses gemma4_e2b_numeric's host oracle — build_references (full-precision "hf"
and IF4-quantized "hostref" HF vision towers) and report (SNR / rel_L2 / max|Δ|)
— but runs the FPGA side through the refactor's Gemma4_UnifiedEngine and
SNR-compares its self._vis_ckpt readbacks at two stages:

  encoder_out   -> reference B  (after all 16 encoder layers, pre-pool)
  image_features-> reference C  (after pooler + embedding projection)

Interpretation (same as numeric.py):
  * FPGA vs HOSTREF (both IF4) should be HIGH SNR — a low value is a real kernel
    bug, not quantization.
  * FPGA vs HF is lower by the IF4 quantization loss (ground-truth gap).

Requires params.bin regenerated with the [LM | vision | host] layout.

Usage:
  python models/gemma4_e2b/gemma4_e2b_numeric_refactor.py            # yosemite, default prompt
  python models/gemma4_e2b/gemma4_e2b_numeric_refactor.py --image people.jpg
  python models/gemma4_e2b/gemma4_e2b_numeric_refactor.py --dev xdma0 --cycle 5.042
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)                                   # gemma4_e2b_refactor / _numeric
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # user_dma_core (repo root)

import user_dma_core
from user_dma_core import set_dma_device
import gemma4_e2b_refactor as g4r
import gemma4_e2b_numeric as num   # reuse build_references() + report() (host oracle)


def main():
    p = argparse.ArgumentParser(
        description="Gemma4 E2B REFACTOR vision-encoder numeric check: host oracle vs FPGA (SNR).")
    p.add_argument("--image", type=str, default=None,
                   help="Image path or bare name in test_samples/ (default: yosemite.jpg).")
    p.add_argument("--prompt", type=str, default="Describe this image in detail.",
                   help="Vision output is prompt-independent; only affects tokenization.")
    p.add_argument("--dev", type=str, default="xdma0")
    p.add_argument("--cycle", type=float, default=1000 / 198.3256)
    args = p.parse_args()

    set_dma_device(args.dev)
    g4r.DMA_DEVICE_H2C = user_dma_core.DMA_DEVICE_H2C
    g4r.DMA_DEVICE_C2H = user_dma_core.DMA_DEVICE_C2H
    g4r.DMA_DEVICE_USER = user_dma_core.DMA_DEVICE_USER
    user_dma_core.CLOCK_CYCLE_TIME_NS = args.cycle
    print(f"Using DMA device: {args.dev}  (cycle {args.cycle:.4f} ns)")

    # Resolve a bare filename against test_samples/, like the refactor's main().
    image_path = args.image or g4r.DEFAULT_IMAGE
    if not os.path.isfile(image_path):
        _cand = os.path.join(os.path.dirname(g4r.DEFAULT_IMAGE), os.path.basename(image_path))
        if os.path.isfile(_cand):
            image_path = _cand
    if not os.path.isfile(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    # --- FPGA side: the refactor engine runs the vision encoder, stashing _vis_ckpt. ---
    print(f"\n[numeric] running REFACTOR FPGA vision encoder on {image_path} ...")
    ue = g4r.Gemma4_UnifiedEngine()
    ue.set_prefill_seq_vlm(image_path, args.prompt)
    ckpt = getattr(ue, "_vis_ckpt", None)
    if not ckpt:
        raise SystemExit("FPGA checkpoints missing (self._vis_ckpt not set).")

    # --- Host side: HF references (hf = ground truth, hostref = IF4/HW-mimicking). ---
    print("\n[numeric] building references ...")
    refs, meta = num.build_references(ue._cfg, image_path, args.prompt)
    real = ~meta["padding"]   # non-padding patches (B is per-patch; C is pooled)

    stages = [("B encoder_out", "B", "encoder_out", real),
              ("C image_features", "C", "image_features", None)]

    print("\n[numeric] ===== FPGA (refactor) vs references — SNR dB (real patches) =====")
    print("  FPGA vs HOSTREF (IF4 — mimics hardware; high = kernel correct):")
    for name, rkey, ckey, mask in stages:
        num.report(name, refs["hostref"][rkey], ckpt[ckey], row_mask=mask)
    print("  FPGA vs HF (full-precision ground truth; gap = IF4 quant loss):")
    for name, rkey, ckey, mask in stages:
        num.report(name, refs["hf"][rkey], ckpt[ckey], row_mask=mask)


if __name__ == "__main__":
    main()
