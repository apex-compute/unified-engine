#!/usr/bin/env python3
"""On-board bring-up of the one-shot VAE phase, one graph prefix at a time.

Compiles a VAE-only program made of the first ``--steps`` ops of the decoder
graph (same planner / emitter as the real program set), runs it on a seeded
random latent and compares the last op's output map with the host IF8
reference intermediate (``reference_decode(..., intermediates=...)``).

    python models/flux2_klein_4b/flux2_klein_4b_vae_probe.py --height 256 --width 256 --engines 8 --steps 2
    python models/flux2_klein_4b/flux2_klein_4b_vae_probe.py --height 256 --width 256 --engines 8 --sweep 1,2,3,4,5,8,12,20,40,80,119
    python models/flux2_klein_4b/flux2_klein_4b_vae_probe.py --height 256 --width 256 --engines 8 --steps all
"""
from __future__ import annotations

import argparse
import math
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
import flux2_klein_4b_vae as V  # noqa: E402

_log = T._log


def compile_prefix(ue: T.Flux2KleinEngine, n_steps: int | None) -> dict:
    """A program set holding only the VAE phase's first ``n_steps`` ops."""
    plan = ue.vae_planner().plan(ue.geo.latent_h, ue.geo.latent_w)
    ue.vae.emit_steps = n_steps
    for eng in ue.E:                          # every prefix program starts at its engine's slot base
        eng._next_program_dram_addr = eng._program_dram_base
    ue._flags_precleared = False              # and with freshly cleared barrier flags
    ue.clear_inst_id()
    ue.start_capture()
    ue.reset_inst_ptr_counter()
    T._SILENT = True
    try:
        ph = ue._emit_phase("vae", ue.vae.emit)
    finally:
        T._SILENT = False
    ue.stop_capture()
    primary = b"".join(inst.get_bytes() for inst in ue.capture_buffer)
    ue.clear_capture_buffer()
    images = [{"engine": 0, "phase": "primary", "addr": ue._program_dram_base, "size": len(primary), "offset": 0}]
    blob = bytearray(primary)
    for wi, (addr, wb) in enumerate(zip(ph[3], ph[4]), start=1):
        images.append({"engine": wi, "phase": "vae", "addr": addr, "size": len(wb), "offset": len(blob)})
        blob.extend(wb)
    meta = {"images": images, "vae_program_start_addr": ph[0], "vae_program_size": ph[1], "vae_flops": ph[2],
            "vae_worker_prog_addrs": ph[3], "vae": plan}
    ue.load_program_images(bytes(blob), meta, check_signature=False)
    for e, eng in enumerate(ue.E):           # worker flag-clear programs are built at run time here
        eng._flag_clear_addr = None
    return meta


def snr_db(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    err = (a - b).pow(2).sum()
    return float("inf") if err == 0 else float(10 * torch.log10(a.pow(2).sum() / err))


def main(argv=None) -> int:
    cfg = T.load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--engines", default="max")
    ap.add_argument("--steps", default="2", help="prefix length, or 'all'")
    ap.add_argument("--sweep", default=None, help="comma-separated prefix lengths to run in turn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", default=cfg["defaults"]["dev"])
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args(argv)

    user_dma_core.set_dma_device(args.dev)
    user_dma_core.configure_clock_from_hardware()
    ne = T.resolve_engines(args.engines, False)
    geo = ref.Flux2Geometry(args.height, args.width, 64, 1)
    paths = T._paths(cfg)
    vae_w = torch.load(paths["vae_weights"], map_location="cpu")
    torch.manual_seed(args.seed)
    latent = torch.randn(cfg["vae"]["latent_channels"], geo.latent_h, geo.latent_w).to(torch.bfloat16)
    inter = {}
    t0 = time.time()
    ref_img = V.reference_decode(vae_w, cfg, latent, intermediates=inter)
    _log(f"host reference: {len(inter)} intermediates in {time.time() - t0:.1f}s")

    ue = T.Flux2KleinEngine(cfg, geo, num_engines=ne)
    ue.timings = {}
    ue.phase_timeout = args.timeout
    plan_steps = ue.vae_planner().plan(geo.latent_h, geo.latent_w)["steps"]
    todo = [int(x) for x in args.sweep.split(",")] if args.sweep else \
        [len(plan_steps) if args.steps == "all" else int(args.steps)]
    ok = True
    for n in todo:
        n = min(n, len(plan_steps))
        meta = compile_prefix(ue, n)
        st = ue.vae.steps[n - 1]
        img = ue.vae.const_image(vae_w)
        assert ue.dma_write(ue.h2c_device, meta["vae"]["const_base"], img, len(img)) == len(img)
        ue.dma_to_accelerator_memory(meta["vae"]["input"], ue.vae.pack_latent(latent))
        t0 = time.time()
        ue._execute_phase(meta, "vae", args.timeout)
        dt = time.time() - t0
        if st["kind"] == "zero_pairs":
            _log(f"[{n:3d}] zero_pairs {dt:.2f}s")
            continue
        if st.get("out_layout") == "tiled":
            flat = ue.dma_from_accelerator_memory(meta["vae"]["output"], (meta["vae"]["output_bytes"] // 2,))
            got = ue.vae.unpack_output(flat)
            want = ref_img
        else:
            got = ue.vae.read_buf(ue, st["out"])
            want = inter[st["name"]]
        s = snr_db(want, got)
        finite = bool(torch.isfinite(got.float()).all())
        ok &= finite and s > 25
        _log(f"[{n:3d}] {st['kind']:9s} {st['name']:52s} {tuple(got.shape)} {dt:6.2f}s  SNR {s:6.1f} dB"
             f"{'' if finite else '  NOT FINITE'}")
        if st["kind"] == "conv" and st.get("out_layout") != "tiled":
            # border check: the producer must leave the halo zero
            b = st["out"]
            full = user_dma_core.packed_map_to_chw(
                ue.dma_from_accelerator_memory(b.addr, (b.lines * 64,)), b.Hp, b.Wp, b.C)
            if b.layout == "padded":
                halo = torch.cat([full[:, 0].flatten(), full[:, -1].flatten(), full[:, :, 0].flatten(), full[:, :, -1].flatten()])
                if halo.float().abs().max() != 0:
                    _log("      WARNING: non-zero halo")
    print("PROBE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
