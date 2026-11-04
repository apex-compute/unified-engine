#!/usr/bin/env python3
"""Check ``flux2_klein_4b_ref.py`` against the official diffusers pipeline.

Needs ``diffusers>=0.37`` and a GPU with >= 12 GB (uses model CPU offload).
Compares prompt embeddings, every per-step latent, and the final image. Run:

    python models/flux2_klein_4b/flux2_klein_4b_compare_diffusers.py --height 256 --width 256 --max-seq-len 128
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flux2_klein_4b_ref as ref  # noqa: E402


def snr_db(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    err = (a - b).pow(2).sum()
    if err == 0:
        return float("inf")
    return float(10 * torch.log10(a.pow(2).sum() / err))


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean()
    return float("inf") if mse == 0 else float(10 * math.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A photo of a red sports car on a mountain road at sunset")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux2_klein_4b_bin"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from diffusers import Flux2KleinPipeline
    snap = ref.snapshot_dir()
    geo = ref.Flux2Geometry(args.height, args.width, args.max_seq_len, args.steps)

    # ---------------- diffusers ----------------
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(snap, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    with torch.no_grad():
        d_embeds, d_ids = pipe.encode_prompt(prompt=args.prompt, device="cuda",
                                             max_sequence_length=args.max_seq_len)
    d_embeds = d_embeds.cpu()
    step_latents = []

    def cb(p, i, t, kw):
        step_latents.append(kw["latents"].detach().cpu().clone())
        return {}

    gen = torch.Generator("cpu").manual_seed(args.seed)
    out = pipe(prompt=args.prompt, height=args.height, width=args.width,
               guidance_scale=1.0, num_inference_steps=args.steps,
               max_sequence_length=args.max_seq_len, generator=gen,
               callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"],
               output_type="pt")
    d_img = (out.images[0].float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu()
    d_sigmas = pipe.scheduler.sigmas.cpu().clone()
    print(f"diffusers done in {time.time() - t0:.0f}s; sigmas={d_sigmas.tolist()}")
    del pipe
    torch.cuda.empty_cache()

    # ---------------- reference ----------------
    t0 = time.time()
    res = ref.run_reference(args.prompt, geo, args.seed, snap=snap, device="cuda", keep_steps=True)
    print(f"reference done in {time.time() - t0:.0f}s; sigmas={res.sigmas.tolist()}")

    # ---------------- compare ----------------
    print(f"sigmas max|diff| = {(res.sigmas - d_sigmas).abs().max():.3e}")
    print(f"prompt_embeds: shape ref={tuple(res.prompt_embeds.shape)} diffusers={tuple(d_embeds.shape)}  "
          f"SNR={snr_db(d_embeds, res.prompt_embeds):.1f} dB  max|diff|={(d_embeds.float() - res.prompt_embeds.float()).abs().max():.3e}")
    for s, (i, sg, v, x) in enumerate(res.steps):
        print(f"step {s} sigma={sg:.4f}: latents SNR vs diffusers = {snr_db(step_latents[s], x):.1f} dB")
    print(f"final latents SNR = {snr_db(step_latents[-1], res.latents):.1f} dB")
    p = psnr(d_img, res.image)
    print(f"image PSNR ref-vs-diffusers = {p:.2f} dB (ref VAE fp32 vs diffusers bf16)")
    ref.save_image(d_img, os.path.join(args.out_dir, "compare_diffusers.png"))
    ref.save_image(res.image, os.path.join(args.out_dir, "compare_reference.png"))
    # Step-0 agreement is ~55 dB (bf16 kernel-order noise); the front-loaded
    # Klein schedule (sigma 1.0 -> 0.70 -> 0) amplifies that noise over the
    # remaining steps, so the latent SNR drifts to ~25 dB while the images stay
    # above 35 dB PSNR. The exact math is checked separately in fp32 by
    # test_flux2_klein_4b.py (single DiT forward and VAE decode vs diffusers).
    ok = snr_db(step_latents[0], res.steps[0][3]) > 45 and snr_db(step_latents[-1], res.latents) > 18 and p > 30
    print("COMPARE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
