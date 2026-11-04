#!/usr/bin/env python3
"""IF4 quantization quality report (milestone M2): BF16 reference vs the
reference run with the exact IF4 weights the board uses (fake-quantized
through quant_lib), per phase.

    python models/flux2_klein_4b/flux2_klein_4b_quant_report.py --height 512 --width 512 --max-seq-len 512

Reports prompt-embedding SNR (Qwen3 IF4), latent SNR and image PSNR / SSIM-ish
agreement, and writes both images next to the bins.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_test as T  # noqa: E402


def main():
    cfg = T.load_config()
    m, dfl = cfg["model"], cfg["defaults"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=dfl["prompt"])
    ap.add_argument("--seed", type=int, default=dfl["seed"])
    ap.add_argument("--height", type=int, default=m["height"])
    ap.add_argument("--width", type=int, default=m["width"])
    ap.add_argument("--max-seq-len", type=int, default=m["max_sequence_length"])
    ap.add_argument("--steps", type=int, default=m["num_inference_steps"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    geo = ref.Flux2Geometry(args.height, args.width, args.max_seq_len, args.steps)
    snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["paths"]["bin_dir"])
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{geo.height}x{geo.width}_L{geo.n_txt}_s{args.seed}"

    bf = ref.run_reference(args.prompt, geo, args.seed, snap=snap, device=args.device, keep_steps=True)
    ref.save_image(bf.image, os.path.join(out_dir, f"quant_bf16_{tag}.png"))
    q = T.quantized_reference(cfg, geo, snap, args.prompt, args.seed, args.device)
    ref.save_image(q.image, os.path.join(out_dir, f"quant_if4_{tag}.png"))
    # IF4 DiT with the BF16 prompt embedding isolates the transformer's share
    q2 = T.quantized_reference(cfg, geo, snap, args.prompt, args.seed, args.device, prompt_embeds=bf.prompt_embeds)

    rep = {
        "geometry": tag, "prompt": args.prompt,
        "embeds_snr_db_if4_te": T.snr_db(bf.prompt_embeds, q.prompt_embeds),
        "latents_snr_db_full_if4": T.snr_db(bf.latents, q.latents),
        "latents_snr_db_if4_dit_only": T.snr_db(bf.latents, q2.latents),
        "image_psnr_db_full_if4": T.psnr_u8(bf.image, q.image),
        "image_psnr_db_if4_dit_only": T.psnr_u8(bf.image, q2.image),
        "step_latent_snr_db_full_if4": [T.snr_db(a[3], b[3]) for a, b in zip(bf.steps, q.steps)],
        "sigmas": bf.sigmas.tolist(),
    }
    print(json.dumps(rep, indent=1))
    with open(os.path.join(out_dir, f"quant_report_{tag}.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
