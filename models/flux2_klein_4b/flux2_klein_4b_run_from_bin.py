#!/usr/bin/env python3
"""Run FLUX.2 [klein] 4B from the single deployable bin.

Everything comes from ONE file (see flux2_klein_4b_pack.py): config, tokenizer,
weights, compiled programs. No network, no HF checkpoint, no other file.

    python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --bin flux2_klein_4b.bin --prompt "..." --out image.png [--config 512x512x64x4]

The bin carries one program set per packed configuration (resolution x text
tokens x steps); ``--config`` picks one (default: the first packed, listed by
``--list``). The engine count is baked in; the board must report at least that many.
"""
from __future__ import annotations

import argparse
import io
import json
import mmap
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import user_dma_core  # noqa: E402
import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_test as T  # noqa: E402
from flux2_klein_4b_pack import read_header  # noqa: E402

_log = T._log


class SingleBin:
    def __init__(self, path: str):
        self.path = path
        self.header, _ = read_header(path)
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def section(self, name: str) -> bytes:
        """A section as in-memory bytes. (The XDMA driver rejects writes straight
        from a file-backed mmap view with EFAULT, so DMA sources are plain bytes.)"""
        s = self.header["sections"][name]
        self.f.seek(s["offset"])
        return self.f.read(s["size"])

    section_bytes = section

    def tokenizer(self):
        from transformers import AutoTokenizer
        tmp = tempfile.mkdtemp(prefix="flux2_tok_")
        for fn in self.header["tokenizer_files"]:
            with open(os.path.join(tmp, fn), "wb") as f:
                f.write(self.section_bytes(f"tokenizer/{fn}"))
        return AutoTokenizer.from_pretrained(tmp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FLUX.2 klein 4B from one bin")
    ap.add_argument("--bin", default=os.path.join(HERE, "flux2_klein_4b_bin", "flux2_klein_4b.bin"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", default="xdma0")
    ap.add_argument("--out", default=None, help="output PNG (default: <bin dir>/flux2_<seed>.png)")
    ap.add_argument("--vae", choices=["fpga", "host"], default="fpga",
                    help="host: decode on the CPU from the bin's IF8 VAE weights (standard FPGA image)")
    ap.add_argument("--vae-timeout", type=float, default=900.0)
    ap.add_argument("--phase-timeout", type=float, default=4 * 3600.0)
    ap.add_argument("--reference", action="store_true",
                    help="also score against the host IF4 reference (needs the HF checkpoint)")
    ap.add_argument("--ref-device", default="cpu")
    ap.add_argument("--config", default=None, help="which packed configuration to run (HxWxTOKENSxSTEPS)")
    ap.add_argument("--list", action="store_true", help="list the packed configurations and exit")
    args = ap.parse_args(argv)

    sb = SingleBin(args.bin)
    h = sb.header
    cfg = h["config"]
    if args.list:
        for k, g in h["configs"].items():
            print(f"  {k}: {g['height']}x{g['width']}, {g['max_sequence_length']} text tokens, {g['num_inference_steps']} steps, {h['engines']} engines")
        return 0
    key = args.config or h["default_config"]
    if key not in h["configs"]:
        raise SystemExit(f"configuration {key} is not in this bin; packed: {list(h['configs'])}")
    g = h["configs"][key]
    geo = ref.Flux2Geometry(g["height"], g["width"], g["max_sequence_length"], g["num_inference_steps"])
    engines = int(h["engines"])
    prompt = args.prompt or cfg["defaults"]["prompt"]
    _log(f"FLUX.2 klein 4B from {os.path.basename(args.bin)} [{key}] | {geo.height}x{geo.width} | text {geo.n_txt} + image "
         f"{geo.n_img} tokens | {geo.num_inference_steps} steps | {engines} engine(s) | built {h.get('built')}")

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    cores = user_dma_core.ANDROMEDA_CORE_COUNT or 1
    if cores < engines:
        raise RuntimeError(f"bin was built for {engines} engines but HW_INFO reports {cores}")
    ue = T.Flux2KleinEngine(cfg, geo, num_engines=engines)
    ue.phase_timeout = args.phase_timeout
    ue.timings = {}
    meta = ue.load_program_images(sb.section(f"programs/{key}"), h["programs_meta"][key], check_signature=False)
    vae_w = torch.load(io.BytesIO(sb.section_bytes("vae_weights")), map_location="cpu")
    ue.load_constants(vae_w)

    # phase 1: text encoder
    tok = sb.tokenizer()
    n_real = int(tok(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False),
                     return_tensors="pt")["input_ids"].shape[-1])
    if n_real > geo.n_txt:
        _log(f"  WARNING: prompt is {n_real} tokens but this configuration keeps {geo.n_txt}; it will be truncated")
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=geo.n_txt)
    te = cfg["text_encoder"]
    embed = torch.frombuffer(bytearray(sb.section_bytes("embed_tokens")), dtype=torch.bfloat16).reshape(te["vocab"], te["hidden"])
    t_all = time.time()
    embeds = ue.run_text_encoder(meta, enc["input_ids"], enc["attention_mask"], embed, sb.section("params_te"))
    _log(f"  TE phase {ue.timings['te_s']:.1f}s")
    del embed

    # phase 2: DiT
    noise = ref.initial_noise(geo, args.seed)
    lat = ue.run_dit(meta, noise, sb.section("params_dit"))
    _log(f"  DiT phase {ue.timings['dit_s']:.1f}s")

    # phase 3: VAE
    latent = ref.unpatchify_latents(ref.unpack_latents(lat, geo.token_h, geo.token_w))
    t0 = time.time()
    if args.vae == "fpga":
        from flux2_klein_4b_vae import Flux2VaeFPGA
        img = Flux2VaeFPGA(ue, vae_w, cfg, timeout_s=args.vae_timeout).decode(latent[0].to(torch.bfloat16))[None]
    else:
        from flux2_klein_4b_vae import reference_decode
        img = reference_decode(vae_w, cfg, latent[0])[None]
    ue.timings["vae_s"] = time.time() - t0
    image = ref.to_uint8_image(img)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.bin)), f"flux2_{args.seed}.png")
    ref.save_image(image, out)
    total = time.time() - t_all
    _log(f"  VAE phase ({args.vae}) {ue.timings['vae_s']:.1f}s -> {out}")
    _log(f"  total {total:.1f}s on the board for {geo.height}x{geo.width}, {geo.num_inference_steps} steps, {engines} engine(s)")

    result = {"height": geo.height, "width": geo.width, "max_sequence_length": geo.n_txt, "steps": geo.num_inference_steps,
              "seed": args.seed, "engines": engines, "image": out, "vae": args.vae, "bin": args.bin, "config": key,
              "finite": bool(torch.isfinite(lat.float()).all()) and bool(torch.isfinite(img.float()).all()),
              "te_s": ue.timings["te_s"], "dit_s": ue.timings["dit_s"], "vae_s": ue.timings["vae_s"], "total_s": total}
    if args.reference:
        snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
        rr = T.quantized_reference(cfg, geo, snap, prompt, args.seed, args.ref_device)
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
