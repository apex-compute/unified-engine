#!/usr/bin/env python3
"""Run FLUX.2 [klein] 4B on the board from the prebuilt files, one configuration per call.

Reads, from ``flux2_klein_4b_bin/``:
  params_te.bin / params_dit.bin / embed_tokens.bin / vae_weights.pt   (flux2_klein_4b_weights.py)
  programs/<HxWxTOKENSxSTEPS>_e<engines>.{bin,json}                     (flux2_klein_4b_test.py --compile-only)
and the tokenizer from the local HF cache. Never quantizes or compiles.

    python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --height 256 --width 256 --max-seq-len 64 --engines 8 --prompt "..." --out car_256.png
    python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --height 512 --width 512 --max-seq-len 64 --engines 8 --prompt "..." --out car_512.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import user_dma_core  # noqa: E402
import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_test as T  # noqa: E402

_log = T._log


FLOPS_PER_CYCLE = 128          # raw peak of one engine (user_dma_core.report_flop_rate_gflops uses the same 128)


def phase_stats(timings: dict, engines: int, total_s: float) -> dict:
    """Per-phase and overall FLOP rates against the raw peak of ``engines`` engines
    (FLOPS_PER_CYCLE each at the clock HW_INFO reported)."""
    clock_hz = float(user_dma_core.CORE_CLOCK_FREQ_HZ or 0.0)
    peak_engine = FLOPS_PER_CYCLE * clock_hz / 1e9                  # GFLOPS of one engine
    peak = peak_engine * engines
    phases = {}
    for name in ("te", "dit", "vae"):
        if f"{name}_gflops" not in timings:
            continue                                        # phase ran on the host
        g = timings[f"{name}_gflops"]
        phases[name] = {"flops": timings[f"{name}_flops"], "device_s": timings[f"{name}_device_s"],
                        "wall_s": timings[f"{name}_s"], "gflops": g, "utilization": g / peak if peak else 0.0}
    total_flops = sum(p["flops"] for p in phases.values())
    device_s = sum(p["device_s"] for p in phases.values())
    total_gflops = total_flops / total_s / 1e9 if total_s else 0.0
    device_gflops = total_flops / device_s / 1e9 if device_s else 0.0
    return {"clock_mhz": clock_hz / 1e6, "peak_gflops": peak, "peak_gflops_per_engine": peak_engine, "phases": phases,
            "total_flops": total_flops, "device_s": device_s, "total_gflops": total_gflops,
            "utilization": total_gflops / peak if peak else 0.0,
            "device_utilization": device_gflops / peak if peak else 0.0}


def main(argv=None) -> int:
    cfg = T.load_config()
    m, dfl = cfg["model"], cfg["defaults"]
    ap = argparse.ArgumentParser(description="FLUX.2 klein 4B, run-only from the prebuilt files")
    ap.add_argument("--prompt", default=dfl["prompt"])
    ap.add_argument("--seed", type=int, default=dfl["seed"])
    ap.add_argument("--height", type=int, default=m["height"])
    ap.add_argument("--width", type=int, default=m["width"])
    ap.add_argument("--max-seq-len", type=int, default=64, help="text tokens the programs were compiled for")
    ap.add_argument("--steps", type=int, default=m["num_inference_steps"])
    ap.add_argument("--engines", default="max", help="engine count or 'max' (HW_INFO core count)")
    ap.add_argument("--dev", default=dfl["dev"])
    ap.add_argument("--out", default=None, help="output PNG (default flux2_klein_4b_bin/flux2_<config>_<seed>.png)")
    ap.add_argument("--vae", choices=["fpga", "host"], default="fpga",
                    help="host: decode on the CPU from the IF8 VAE weights (standard FPGA image)")
    ap.add_argument("--phase-timeout", type=float, default=4 * 3600.0)
    ap.add_argument("--reference", action="store_true",
                    help="also score on the host: the decoder vs the IF8 oracle and the whole pipeline vs "
                         "the IF4 reference (needs the HF checkpoint); off by default, nothing runs on the CPU")
    ap.add_argument("--ref-device", default="cpu")
    args = ap.parse_args(argv)

    geo = ref.Flux2Geometry(args.height, args.width, args.max_seq_len, args.steps)
    paths = T._paths(cfg)
    for k in ("params_te_bin", "params_dit_bin", "vae_weights", "embed_tokens_bin"):
        if not os.path.exists(paths[k]):
            raise FileNotFoundError(f"{paths[k]} missing: run flux2_klein_4b_weights.py first")
    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    ne = T.resolve_engines(args.engines, False)
    key = T.config_key(geo)
    _log(f"FLUX.2 klein 4B [{key}] | {geo.height}x{geo.width} | text {geo.n_txt} + image {geo.n_img} tokens | "
         f"{geo.num_inference_steps} steps | {ne} engine(s)")

    ue = T.Flux2KleinEngine(cfg, geo, num_engines=ne)
    ue.phase_timeout = args.phase_timeout
    ue.timings = {}
    meta = T.ensure_programs(ue, cfg, allow_compile=False)
    bin_p, meta_p = T.program_paths(cfg, geo, ne)
    ue.load_programs(bin_p, meta_p)
    vae_w = torch.load(paths["vae_weights"], map_location="cpu")
    ue.load_constants(vae_w)

    # phase 1: text encoder
    snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
    input_ids, attention_mask = T.tokenize(snap, args.prompt, geo.n_txt)
    n_real = int(T.tokenize(snap, args.prompt, 4096)[1].sum())
    if n_real > geo.n_txt:
        _log(f"  WARNING: prompt is {n_real} tokens but this configuration keeps {geo.n_txt}; it is truncated")
    embed_tokens = T.load_embed_tokens(cfg, paths)
    t_all = time.time()
    embeds = ue.run_text_encoder(meta, input_ids, attention_mask, embed_tokens, paths["params_te_bin"])
    _log(f"  TE phase {ue.timings['te_s']:.1f}s")
    del embed_tokens

    # phase 2: DiT
    noise = ref.initial_noise(geo, args.seed)
    lat = ue.run_dit(meta, noise, paths["params_dit_bin"])
    _log(f"  DiT phase {ue.timings['dit_s']:.1f}s")

    # phase 3: VAE
    latent = ref.unpatchify_latents(ref.unpack_latents(lat, geo.token_h, geo.token_w))
    if args.vae == "fpga":
        img = ue.run_vae(meta, latent[0].to(torch.bfloat16), vae_w)[None]
    else:
        from flux2_klein_4b_vae import reference_decode
        t0 = time.time()
        img = reference_decode(vae_w, cfg, latent[0])[None]
        ue.timings["vae_s"] = time.time() - t0
    image = ref.to_uint8_image(img)
    out = args.out or os.path.join(HERE, cfg["paths"]["bin_dir"], f"flux2_{key}_{args.seed}.png")
    ref.save_image(image, out)
    total = time.time() - t_all
    _log(f"  VAE phase ({args.vae}) {ue.timings['vae_s']:.1f}s -> {out}")
    _log(f"  total {total:.1f}s on the board for {geo.height}x{geo.width}, {geo.num_inference_steps} steps, {ne} engine(s)")

    result = {"height": geo.height, "width": geo.width, "max_sequence_length": geo.n_txt, "steps": geo.num_inference_steps,
              "seed": args.seed, "engines": ne, "image": out, "vae": args.vae, "config": key,
              "finite": bool(torch.isfinite(lat.float()).all()) and bool(torch.isfinite(img.float()).all()),
              "te_s": ue.timings["te_s"], "dit_s": ue.timings["dit_s"], "vae_s": ue.timings["vae_s"], "total_s": total}
    result.update(phase_stats(ue.timings, ne, total))
    for name, ph in result["phases"].items():
        _log(f"  {name.upper()}: {ph['flops'] / 1e12:.2f} TFLOP in {ph['device_s']:.1f}s on the device = "
             f"{ph['gflops']:.0f} GFLOPS, {100 * ph['utilization']:.0f}% of the {result['peak_gflops']:.0f} GFLOPS peak")
    _log(f"  overall: {result['total_flops'] / 1e12:.2f} TFLOP / {total:.1f}s = {result['total_gflops']:.0f} GFLOPS, "
         f"{100 * result['utilization']:.0f}% of peak ({100 * result['device_utilization']:.0f}% on the device, "
         f"{ne} engine(s) at {result['clock_mhz']:.0f} MHz)")
    if args.vae == "fpga" and args.reference:
        from flux2_klein_4b_vae import reference_decode
        oracle = reference_decode(vae_w, cfg, latent[0])[None]
        result["vae_psnr_db"] = T.psnr_u8(ref.to_uint8_image(oracle), image)
        _log(f"  FPGA decoder vs IF8 host oracle: image PSNR {result['vae_psnr_db']:.1f} dB")
    if args.reference:
        rr = T.quantized_reference(cfg, geo, snap, args.prompt, args.seed, args.ref_device)
        result["embeds_snr_db"] = T.snr_db(rr.prompt_embeds, embeds)
        result["step_latents_snr_db"] = [T.snr_db(x, ue.step_latents[s + 1]) for s, (i, sg, v, x) in enumerate(rr.steps)]
        result["latents_snr_db"] = T.snr_db(rr.latents, ue.latents_raw)
        result["image_psnr_db"] = T.psnr_u8(rr.image, image)
        _log(f"  vs IF4 host reference: embeds {result['embeds_snr_db']:.1f} dB, per-step latents "
             f"{[round(x, 1) for x in result['step_latents_snr_db']]} dB, image PSNR {result['image_psnr_db']:.1f} dB")
    print("TEST_RESULT: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
