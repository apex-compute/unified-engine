#!/usr/bin/env python3
"""Run FLUX.2 [klein] 4B on the board from the prebuilt files.

Reads, from ``flux2_klein_4b_bin/``:
  params_te.bin / params_dit.bin / embed_tokens.bin / vae_weights.pt   (flux2_klein_4b_weights.py)
  programs/<HxWxTOKENSxSTEPS>_e<engines>.{bin,json}                     (flux2_klein_4b_test.py --compile-only)
and the tokenizer from the local HF cache. Never quantizes or compiles.

One image per call:
    python models/flux2_klein_4b/flux2_klein_4b_run_from_bin.py --height 256 --width 256 --max-seq-len 64 --engines 8 --prompt "..." --out car_256.png

Many images per process (``--serve``, what the web UI's "keep the board loaded" mode drives):
one JSON job per stdin line, ``TEST_RESULT: {...}`` or ``JOB_ERROR: {...}`` per job on stdout.
The process keeps the device, the tokenizer, the VAE weights and the program set of the last
configuration; a job of the same configuration skips the program DMA. The 778 MB BF16 token
embedding table lives on the board (top of the tensor region, nothing overwrites it) and is sent
once per process; the TE program gathers the prompt's rows from it. The TE / DiT weight images
are re-sent on every run regardless: they and the VAE arena time-share the params arena
(everything sits below the 4 GiB DMA window), so a run always overwrites what the previous one
left there.
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
    (FLOPS_PER_CYCLE each at the clock HW_INFO reported), from the engines' own execution
    time only; the host side of a phase (weight DMA, tokenizing) is not in the rates."""
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
    device_s = sum(p["device_s"] for p in phases.values())        # engine time only: no host side
    total_gflops = total_flops / device_s / 1e9 if device_s else 0.0
    return {"clock_mhz": clock_hz / 1e6, "peak_gflops": peak, "peak_gflops_per_engine": peak_engine, "phases": phases,
            "total_flops": total_flops, "device_s": device_s, "total_gflops": total_gflops,
            "utilization": total_gflops / peak if peak else 0.0, "wall_s": total_s}


class Session:
    """The board plus everything a run needs from the host, kept across runs."""

    def __init__(self, cfg: dict, dev: str, engines: str, phase_timeout: float):
        self.cfg = cfg
        self.paths = T._paths(cfg)
        for k in ("params_te_bin", "params_dit_bin", "vae_weights", "embed_tokens_bin"):
            if not os.path.exists(self.paths[k]):
                raise FileNotFoundError(f"{self.paths[k]} missing: run flux2_klein_4b_weights.py first")
        user_dma_core.set_dma_device(dev)
        user_dma_core.configure_clock_from_hardware()
        self.ne = T.resolve_engines(engines, False)
        self.phase_timeout = phase_timeout
        t0 = time.time()
        self.snap = ref.snapshot_dir(cfg["paths"]["hf_model_repo"])
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(os.path.join(self.snap, "tokenizer"))
        self.vae_w = torch.load(self.paths["vae_weights"], map_location="cpu")
        self.host_load_s = time.time() - t0
        self.engines: dict[str, tuple] = {}        # config key -> (engine, program meta)
        self.loaded: str | None = None             # config key whose programs + constants are on the board
        self.embed_resident = False                # the token embedding table has been sent to the board
        _log(f"  host state loaded in {self.host_load_s:.1f}s (tokenizer, VAE weights); "
             f"{self.ne} engine(s) at {float(user_dma_core.CORE_CLOCK_FREQ_HZ or 0) / 1e6:.0f} MHz")

    def tokenize(self, prompt: str, L: int):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        enc = self.tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=L)
        return enc["input_ids"], enc["attention_mask"]

    def engine(self, geo: ref.Flux2Geometry):
        key = T.config_key(geo)
        if key not in self.engines:
            ue = T.Flux2KleinEngine(self.cfg, geo, num_engines=self.ne)
            ue.phase_timeout = self.phase_timeout
            meta = T.ensure_programs(ue, self.cfg, allow_compile=False)
            self.engines[key] = (ue, meta)
        return key, self.engines[key][0], self.engines[key][1]

    def run(self, prompt: str, seed: int, geo: ref.Flux2Geometry, out: str, *, vae: str = "fpga",
            reload: bool = True) -> tuple[dict, dict]:
        """One image. Returns (result, aux) where aux holds the tensors a host reference is scored against."""
        key, ue, meta = self.engine(geo)
        ue.timings = {}
        _log(f"FLUX.2 klein 4B [{key}] | {geo.height}x{geo.width} | text {geo.n_txt} + image {geo.n_img} tokens | "
             f"{geo.num_inference_steps} steps | {self.ne} engine(s)")
        t_setup = time.time()
        programs_loaded = reload or self.loaded != key
        if programs_loaded:
            bin_p, meta_p = T.program_paths(self.cfg, geo, self.ne)
            ue.load_programs(bin_p, meta_p)
            ue.load_constants(self.vae_w)
            self.loaded = key
        else:
            _log(f"    programs and constants of [{key}] already on the board (no reload)")
        embed_sent = not self.embed_resident
        if embed_sent:                              # the table sits above every tensor: once per process
            ue.load_embed_table(self.paths["embed_tokens_bin"])
            self.embed_resident = True
        else:
            _log("    token embedding table already on the board")
        input_ids, attention_mask = self.tokenize(prompt, geo.n_txt)
        n_real = int(self.tokenize(prompt, 4096)[1].sum())
        if n_real > geo.n_txt:
            _log(f"  WARNING: prompt is {n_real} tokens but this configuration keeps {geo.n_txt}; it is truncated")
        setup_s = time.time() - t_setup

        t_all = time.time()
        embeds = ue.run_text_encoder(meta, input_ids, attention_mask, self.paths["params_te_bin"])
        _log(f"  TE phase {ue.timings['te_s']:.1f}s")
        noise = ref.initial_noise(geo, seed)
        lat = ue.run_dit(meta, noise, self.paths["params_dit_bin"])
        _log(f"  DiT phase {ue.timings['dit_s']:.1f}s")
        latent = ref.unpatchify_latents(ref.unpack_latents(lat, geo.token_h, geo.token_w))
        if vae == "fpga":
            img = ue.run_vae(meta, latent[0].to(torch.bfloat16), self.vae_w)[None]
        else:
            from flux2_klein_4b_vae import reference_decode
            t0 = time.time()
            img = reference_decode(self.vae_w, self.cfg, latent[0])[None]
            ue.timings["vae_s"] = time.time() - t0
        image = ref.to_uint8_image(img)
        ref.save_image(image, out)
        total = time.time() - t_all
        _log(f"  VAE phase ({vae}) {ue.timings['vae_s']:.1f}s -> {out}")
        _log(f"  total {total:.1f}s on the board for {geo.height}x{geo.width}, {geo.num_inference_steps} steps, "
             f"{self.ne} engine(s); setup {setup_s:.1f}s ({'programs reloaded' if programs_loaded else 'programs kept'})")

        weight_bytes = os.path.getsize(self.paths["params_te_bin"]) + os.path.getsize(self.paths["params_dit_bin"])
        embed_bytes = os.path.getsize(self.paths["embed_tokens_bin"])
        result = {"height": geo.height, "width": geo.width, "max_sequence_length": geo.n_txt,
                  "steps": geo.num_inference_steps, "seed": seed, "engines": self.ne, "image": out, "vae": vae,
                  "config": key, "prompt_tokens": n_real,
                  "finite": bool(torch.isfinite(lat.float()).all()) and bool(torch.isfinite(img.float()).all()),
                  "te_s": ue.timings["te_s"], "dit_s": ue.timings["dit_s"], "vae_s": ue.timings["vae_s"],
                  "total_s": total, "setup_s": setup_s,
                  "reload": {"programs": programs_loaded, "weights": True, "weight_bytes": weight_bytes,
                             "embed": embed_sent, "embed_bytes": embed_bytes}}
        result.update(phase_stats(ue.timings, self.ne, total))
        for name, ph in result["phases"].items():
            _log(f"  {name.upper()}: {ph['flops'] / 1e12:.2f} TFLOP in {ph['device_s']:.1f}s on the device = "
                 f"{ph['gflops']:.0f} GFLOPS, {100 * ph['utilization']:.0f}% of the {result['peak_gflops']:.0f} GFLOPS peak")
        _log(f"  overall: {result['total_flops'] / 1e12:.2f} TFLOP / {result['device_s']:.1f}s of engine time = "
             f"{result['total_gflops']:.0f} GFLOPS, {100 * result['utilization']:.0f}% of peak "
             f"({self.ne} engine(s) at {result['clock_mhz']:.0f} MHz; {total:.1f}s wall with the host side)")
        aux = {"embeds": embeds, "latent": latent, "image": image, "ue": ue}
        return result, aux


def serve(session: Session, cfg: dict) -> int:
    """Job loop for the web UI: one JSON object per stdin line, one TEST_RESULT / JOB_ERROR line per job."""
    sys.stdout.reconfigure(line_buffering=True)
    m = cfg["model"]
    print("SERVE_READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            geo = ref.Flux2Geometry(int(job.get("height", m["height"])), int(job.get("width", m["width"])),
                                    int(job.get("max_seq_len", 64)), int(job.get("steps", m["num_inference_steps"])))
            out = job.get("out") or os.path.join(HERE, cfg["paths"]["bin_dir"],
                                                 f"flux2_{T.config_key(geo)}_{int(job.get('seed', 0))}.png")
            result, _ = session.run(str(job["prompt"]), int(job.get("seed", 0)), geo, out,
                                    vae=job.get("vae", "fpga"), reload=bool(job.get("reload", False)))
            print("TEST_RESULT: " + json.dumps(result), flush=True)
        except Exception as e:  # noqa: BLE001 - the loop must survive a bad job
            import traceback
            traceback.print_exc()
            print("JOB_ERROR: " + json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
    return 0


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
    ap.add_argument("--serve", action="store_true",
                    help="keep the board loaded and run one job per stdin line (JSON); see the module docstring")
    ap.add_argument("--reference", action="store_true",
                    help="also score on the host: the decoder vs the IF8 oracle and the whole pipeline vs "
                         "the IF4 reference (needs the HF checkpoint); off by default, nothing runs on the CPU")
    ap.add_argument("--ref-device", default="cpu")
    args = ap.parse_args(argv)

    session = Session(cfg, args.dev, args.engines, args.phase_timeout)
    if args.serve:
        return serve(session, cfg)

    geo = ref.Flux2Geometry(args.height, args.width, args.max_seq_len, args.steps)
    out = args.out or os.path.join(HERE, cfg["paths"]["bin_dir"], f"flux2_{T.config_key(geo)}_{args.seed}.png")
    result, aux = session.run(args.prompt, args.seed, geo, out, vae=args.vae, reload=True)
    if args.vae == "fpga" and args.reference:
        from flux2_klein_4b_vae import reference_decode
        oracle = reference_decode(session.vae_w, cfg, aux["latent"][0])[None]
        result["vae_psnr_db"] = T.psnr_u8(ref.to_uint8_image(oracle), aux["image"])
        _log(f"  FPGA decoder vs the host oracle ({cfg['quantization'].get('vae_precision', 'if8')} weights): "
             f"image PSNR {result['vae_psnr_db']:.1f} dB")
    if args.reference:
        ue = aux["ue"]
        rr = T.quantized_reference(cfg, geo, session.snap, args.prompt, args.seed, args.ref_device)
        result["embeds_snr_db"] = T.snr_db(rr.prompt_embeds, aux["embeds"])
        result["step_latents_snr_db"] = [T.snr_db(x, ue.step_latents[s + 1]) for s, (i, sg, v, x) in enumerate(rr.steps)]
        result["latents_snr_db"] = T.snr_db(rr.latents, ue.latents_raw)
        result["image_psnr_db"] = T.psnr_u8(rr.image, aux["image"])
        _log(f"  vs IF4 host reference: embeds {result['embeds_snr_db']:.1f} dB, per-step latents "
             f"{[round(x, 1) for x in result['step_latents_snr_db']]} dB, image PSNR {result['image_psnr_db']:.1f} dB")
    print("TEST_RESULT: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
